#!/usr/bin/env python3
"""Initialize the exhaustive, fail-closed comprehensive-report worksheet."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from myrec.mechanism.comprehensive_report_builder import (
    build_comprehensive_decision_template,
    populate_registered_component_model_coverage,
)
from myrec.mechanism.deep_dive_closeout_audit import EXPECTED_DELIVERABLES
from myrec.mechanism.supplemental_evidence_registry import (
    audit_supplemental_evidence_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="experiments/motivation/transformer_comprehensive_decisions.json",
    )
    parser.add_argument(
        "--report-id", default="motivation_transformer_comprehensive_v1"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite worksheet: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_comprehensive_decision_template(report_id=args.report_id)
    supplement_audit = audit_supplemental_evidence_registry(".")
    supplement_metadata = {
        str(row["evidence_id"]): row for row in supplement_audit["entries"]
    }
    payload = populate_registered_component_model_coverage(
        payload,
        registered_formal=set(EXPECTED_DELIVERABLES),
        registered_supplements=supplement_metadata,
    )
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    main()
