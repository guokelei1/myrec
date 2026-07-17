from __future__ import annotations

import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import yaml
from myrec.utils.hashing import sha256_file

from myrec.baselines.motivation_v12_ranker import (
    _assert_scoring_population,
    _base_inference_artifact_identity,
    _build_training_contract,
    _enforce_score_nondegeneracy,
    _mean_target_sequence_nll,
    _restore_rng_state,
    _runtime_metadata,
    _save_training_checkpoint,
    _score_instructrec_request,
    _target_sequence_log_likelihoods,
    listwise_softmax_loss,
    load_v12_ranker_config,
    pairwise_ranknet_loss,
)


class _ConstantLogitModel:
    def __call__(self, input_ids, attention_mask, use_cache, logits_to_keep):
        del attention_mask, use_cache
        logits = torch.tensor([0.0, 1.0, 2.0, 3.0], device=input_ids.device)
        values = logits.view(1, 1, 4).expand(
            input_ids.shape[0], logits_to_keep, 4
        )
        return type("Output", (), {"logits": values})()


class _PositionCausalModel:
    def __call__(self, input_ids, attention_mask, use_cache, logits_to_keep):
        del attention_mask, use_cache
        batch, width = input_ids.shape
        values = torch.full((batch, logits_to_keep, 16), -20.0)
        start = width - logits_to_keep
        for row in range(batch):
            for local, position in enumerate(range(start, width)):
                if position + 1 < width:
                    values[row, local, int(input_ids[row, position + 1])] = 20.0
        return type("Output", (), {"logits": values})()


class _RepeatableCache:
    def __init__(self):
        self.batch_size = 1

    def batch_repeat_interleave(self, repeats):
        self.batch_size *= repeats


class _CountingCachedCausalModel:
    def __init__(self, vocabulary_size=32):
        self.vocabulary_size = vocabulary_size
        self.prefix_calls = 0
        self.continuation_calls = 0

    def __call__(
        self,
        input_ids,
        attention_mask,
        use_cache,
        logits_to_keep=None,
        past_key_values=None,
    ):
        del attention_mask
        if past_key_values is None and use_cache:
            self.prefix_calls += 1
        elif past_key_values is not None:
            self.continuation_calls += 1
        width = input_ids.shape[1] if logits_to_keep is None else logits_to_keep
        source = input_ids[:, -width:]
        vocabulary = torch.arange(
            self.vocabulary_size, device=input_ids.device, dtype=torch.float32
        )
        preferred = (source + 1).remainder(self.vocabulary_size)
        logits = -(vocabulary.view(1, 1, -1) - preferred[:, :, None]).abs()
        return SimpleNamespace(
            logits=logits,
            past_key_values=_RepeatableCache() if use_cache else None,
        )


class _TinySaveableModel(torch.nn.Linear):
    def save_pretrained(self, path, safe_serialization=True):
        self.saved_safely = safe_serialization
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "tiny.pt")
        (path / "model.safetensors").write_bytes(b"checkpoint-identity-fixture")


class _TinyTokenizer:
    def save_pretrained(self, path):
        Path(path, "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


class MotivationV12RankerTest(unittest.TestCase):
    def test_base_inference_identity_binds_runtime_configs_and_vocab(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            (root / "config.json").write_text('{"layers": 1}\n', encoding="utf-8")
            (root / "README.md").write_text("not runtime\n", encoding="utf-8")
            first = _base_inference_artifact_identity(root)
            self.assertEqual(
                [row["name"] for row in first["files"]],
                ["config.json", "model.safetensors", "tokenizer.json"],
            )
            (root / "README.md").write_text("changed docs\n", encoding="utf-8")
            self.assertEqual(
                _base_inference_artifact_identity(root)["sha256"], first["sha256"]
            )
            (root / "config.json").write_text('{"layers": 2}\n', encoding="utf-8")
            self.assertNotEqual(
                _base_inference_artifact_identity(root)["sha256"], first["sha256"]
            )

    def test_scoring_population_rejects_unregistered_dataset_version(self):
        config = {
            "_protocol": {
                "data": {"development_population": {"dataset_version": "dev-v1"}}
            }
        }
        with self.assertRaisesRegex(ValueError, "unregistered"):
            _assert_scoring_population(
                Path("unused"),
                config,
                split="confirmation",
                dataset_manifest={"dataset_version": "tampered-v2"},
                checkpoint_id="checkpoint",
                checkpoint_weight_files=[
                    {"name": "model", "sha256": "a", "size_bytes": 1}
                ],
                training_metadata_path=Path("unused-metadata"),
            )

    def test_holdout_population_binds_actual_checkpoint_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = root / "training_metadata.json"
            metadata_path.write_text("{}\n", encoding="utf-8")
            files = [
                {"name": "model.safetensors", "sha256": "a" * 64, "size_bytes": 7}
            ]
            config = {
                "_config_sha256": "b" * 64,
                "_protocol": {
                    "data": {
                        "development_population": {"dataset_version": "dev-v1"}
                    }
                },
                "method_id": "q0_qwen3_reranker_06b",
                "protocol": {"path": "protocol.yaml", "sha256": "c" * 64},
            }
            frozen = {
                "checkpoint_id": "checkpoint-fixed",
                "checkpoint_files": [{**files[0], "path": "ignored"}],
                "config_sha256": config["_config_sha256"],
                "identity_manifest_sha256": "d" * 64,
                "implementation_digest": "e" * 64,
                "protocol_sha256": config["protocol"]["sha256"],
                "training_metadata_sha256": sha256_file(metadata_path),
            }
            audit = {
                "checkpoint_identities": {config["method_id"]: frozen},
                "integrity_lock_sha256": "f" * 64,
                "manifest_sha256": "1" * 64,
                "post_selection_recipe_checkpoint_lock_sha256": "2" * 64,
                "protocol_sha256": config["protocol"]["sha256"],
            }
            with patch(
                "myrec.baselines.motivation_v12_ranker.verify_published_holdout",
                return_value=audit,
            ), patch(
                "myrec.baselines.motivation_v12_ranker._implementation_identity",
                return_value={"digest": frozen["implementation_digest"]},
            ):
                observed = _assert_scoring_population(
                    root,
                    config,
                    split="confirmation",
                    dataset_manifest={
                        "dataset_version": "full_confirm_preceding40k_newholdout4k_v12"
                    },
                    checkpoint_id=frozen["checkpoint_id"],
                    checkpoint_weight_files=files,
                    training_metadata_path=metadata_path,
                )
                self.assertFalse(observed["qrels_opened"])
                with self.assertRaisesRegex(ValueError, "checkpoint_id"):
                    _assert_scoring_population(
                        root,
                        config,
                        split="confirmation",
                        dataset_manifest={
                            "dataset_version": "full_confirm_preceding40k_newholdout4k_v12"
                        },
                        checkpoint_id="wrong-checkpoint",
                        checkpoint_weight_files=files,
                        training_metadata_path=metadata_path,
                    )

    def test_uncapped_globally_degenerate_score_run_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "scores.jsonl").write_text("partial\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "globally degenerate"):
                _enforce_score_nondegeneracy(
                    run_dir,
                    [0.0, 1.0e-9, 1.0e-8],
                    capped=False,
                )
            self.assertFalse(run_dir.exists())

    def test_runtime_metadata_records_python_and_q3_peft(self):
        torch_module = SimpleNamespace(__version__="torch-test")
        transformers_module = SimpleNamespace(__version__="transformers-test")
        common = _runtime_metadata(
            "q0_qwen3_reranker_06b", torch_module, transformers_module
        )
        self.assertEqual(common["python_executable"], sys.executable)
        self.assertTrue(common["python_version"])
        self.assertEqual(
            common["package_versions"],
            {"torch": "torch-test", "transformers": "transformers-test"},
        )
        with patch.dict(sys.modules, {"peft": SimpleNamespace(__version__="peft-test")}):
            q3 = _runtime_metadata(
                "q3_tallrec_generalqwen", torch_module, transformers_module
            )
        self.assertEqual(q3["package_versions"]["peft"], "peft-test")

    def test_resume_training_contract_is_bound_to_implementation_digest(self):
        values = {
            "batches_per_epoch": 4,
            "config_sha256": "config",
            "evidence_mode": "first_round_pilot",
            "group_count": 8,
            "implementation_digest": "implementation-a",
            "manifest_sha256": "manifest",
            "max_train_groups": None,
            "qrels_train_sha256": "qrels",
            "records_train_sha256": "records",
            "total_optimizer_steps": 2,
            "updates_per_epoch": 2,
        }
        original = _build_training_contract(**values)
        drifted = _build_training_contract(
            **{**values, "implementation_digest": "implementation-b"}
        )
        self.assertEqual(original["implementation_digest"], "implementation-a")
        self.assertNotEqual(original, drifted)

    def test_ranknet_and_listnet_match_hand_computation(self):
        scores = torch.tensor([2.0, 1.0, 0.0])
        pair = pairwise_ranknet_loss(scores, [2.0, 1.0, 0.0])
        expected_pair = sum(
            math.log1p(math.exp(-margin)) for margin in (1.0, 2.0, 1.0)
        ) / 3.0
        self.assertAlmostEqual(float(pair), expected_pair, places=6)

        listwise = listwise_softmax_loss(scores, [2.0, 1.0, 0.0])
        log_probs = torch.log_softmax(scores, dim=0)
        expected_listwise = -(0.75 * log_probs[0] + 0.25 * log_probs[1])
        self.assertAlmostEqual(float(listwise), float(expected_listwise), places=6)

    def test_sequence_likelihood_and_output_only_nll_are_length_normalized(self):
        model = _ConstantLogitModel()
        prompts = [[3, 2], [1]]
        targets = [[1, 2], [3]]
        observed = _target_sequence_log_likelihoods(
            model, prompts, targets, pad_token_id=0, device="cpu"
        )
        log_probs = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0, 3.0]), dim=0)
        expected = [float((log_probs[1] + log_probs[2]) / 2), float(log_probs[3])]
        self.assertEqual(len(observed), 2)
        for left, right in zip(observed, expected):
            self.assertAlmostEqual(left, right, places=6)
        nll = _mean_target_sequence_nll(
            model, prompts, targets, pad_token_id=0, device="cpu"
        )
        self.assertAlmostEqual(float(nll), -sum(expected) / 2, places=6)

    def test_target_slicing_uses_each_causal_predecessor_position(self):
        prompts = [[9, 8, 7], [4]]
        targets = [[1, 2, 3], [5]]
        values = _target_sequence_log_likelihoods(
            _PositionCausalModel(),
            prompts,
            targets,
            pad_token_id=0,
            device="cpu",
        )
        self.assertTrue(all(value > -1.0e-6 for value in values))

    def test_q1_reuses_one_exact_prefix_cache_across_candidate_chunks(self):
        candidates = tuple({"item_id": str(index)} for index in range(5))
        record = SimpleNamespace(request_id="request-1", candidates=candidates)
        prompt = [3, 5]
        targets = {
            str(index): [6, 7 + index, 8 + index] for index in range(len(candidates))
        }
        config = {
            "training": {
                "context_token_budget": 8,
                "history_budget": 6,
                "max_length": 64,
                "max_target_length": 8,
                "seed": 17,
            }
        }
        model = _CountingCachedCausalModel()
        tokenizer = SimpleNamespace(pad_token_id=0)
        with patch(
            "myrec.baselines.motivation_v12_ranker.encode_instructrec_selection_prompt",
            return_value=(prompt, targets, {}),
        ):
            observed, at_boundary = _score_instructrec_request(
                model,
                tokenizer,
                record,
                [],
                config,
                device="cpu",
                batch_size=2,
            )
        expected_model = _CountingCachedCausalModel()
        expected = {
            item_id: _target_sequence_log_likelihoods(
                expected_model,
                [prompt],
                [target],
                pad_token_id=0,
                device="cpu",
            )[0]
            for item_id, target in targets.items()
        }
        self.assertFalse(at_boundary)
        self.assertEqual(model.prefix_calls, 1)
        self.assertEqual(model.continuation_calls, 3)
        self.assertEqual(set(observed), set(expected))
        for item_id in expected:
            self.assertAlmostEqual(observed[item_id], expected[item_id], places=6)

    def test_checkpoint_resume_restores_model_optimizer_scheduler_rng_and_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            random.seed(91)
            torch.manual_seed(91)
            model = _TinySaveableModel(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: 1.0 / (step + 1)
            )
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            _tiny_update(model, optimizer, scheduler)
            progress = {
                "epoch": 0,
                "batch_cursor": 7,
                "micro_steps": 7,
                "optimizer_steps": 1,
            }
            config = {
                "_config_sha256": "frozen-config",
                "method_id": "q0_qwen3_reranker_06b",
            }
            _save_training_checkpoint(
                root,
                model,
                _TinyTokenizer(),
                optimizer,
                scheduler,
                scaler,
                progress,
                config,
                {"fixture": "resume-contract"},
                torch,
            )

            baseline_python = random.random()
            baseline_torch = torch.rand(2)
            _tiny_update(model, optimizer, scheduler)
            baseline_parameters = [value.detach().clone() for value in model.parameters()]
            baseline_lr = scheduler.get_last_lr()

            state = torch.load(
                root / "checkpoint_latest" / "trainer_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            resumed = _TinySaveableModel(2, 1)
            resumed.load_state_dict(
                torch.load(
                    root / "checkpoint_latest" / "model" / "tiny.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
            resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
                resumed_optimizer, lambda step: 1.0 / (step + 1)
            )
            resumed_scaler = torch.amp.GradScaler("cuda", enabled=False)
            resumed_optimizer.load_state_dict(state["optimizer"])
            resumed_scheduler.load_state_dict(state["scheduler"])
            resumed_scaler.load_state_dict(state["scaler"])
            _restore_rng_state(torch, state["rng"])
            self.assertEqual(state["progress"], progress)
            self.assertEqual(random.random(), baseline_python)
            torch.testing.assert_close(torch.rand(2), baseline_torch)
            _tiny_update(resumed, resumed_optimizer, resumed_scheduler)
            for expected, actual in zip(baseline_parameters, resumed.parameters()):
                torch.testing.assert_close(actual, expected)
            self.assertEqual(resumed_scheduler.get_last_lr(), baseline_lr)

    def test_real_configs_are_protocol_locked_and_recipe_drift_is_rejected(self):
        repo = Path(__file__).resolve().parents[1]
        config_path = (
            repo
            / "configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
        )
        loaded = load_v12_ranker_config(config_path)
        self.assertEqual(loaded["training"]["seed"], 20260714)
        with tempfile.TemporaryDirectory() as tmp:
            drifted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            drifted["training"]["learning_rate"] = 9.0e-4
            drifted_path = Path(tmp) / "drifted.yaml"
            drifted_path.write_text(
                yaml.safe_dump(drifted, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "learning_rate drifted"):
                load_v12_ranker_config(drifted_path)


def _tiny_update(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    features = torch.randn(3, 2)
    multiplier = random.random()
    loss = (model(features).square().mean()) * multiplier
    loss.backward()
    optimizer.step()
    scheduler.step()


if __name__ == "__main__":
    unittest.main()
