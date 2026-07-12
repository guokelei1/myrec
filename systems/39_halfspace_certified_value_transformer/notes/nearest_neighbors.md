# C39 nearest-neighbor audit

Pre-outcome verdict: **distinct-with-uncertainty**. The search found no paper
with the exact candidate/event pre-aggregation score-halfspace value operator,
but global novelty is not established and cannot be claimed from this audit.

| Neighbor | What it covers | C39 falsifiable boundary |
|---|---|---|
| [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need) | Candidate/query-dependent weights over candidate-independent values | `eventwise_raw` removes only C39's value projection |
| [Dynamic Filter Networks](https://papers.nips.cc/paper_files/paper/2016/hash/8bf1211fd4b7b94528899de0a43b9fb3-Abstract.html) | Input-conditioned generated filters | C39 has no learned filter generator; its pair map is the unique closed-form halfspace projection |
| [Dynamic Edge-Conditioned Filters](https://openaccess.thecvf.com/content_cvpr_2017/html/Simonovsky_Dynamic_Edge-Conditioned_Filters_CVPR_2017_paper.html) | Per-edge learned filters before neighbor aggregation | A generic edge MLP can imitate C39 but does not enforce its nonnegative readout law; `eventwise_raw` is the unconstrained degeneration |
| [FiLM](https://ojs.aaai.org/index.php/AAAI/article/view/11671) | Feature-wise affine conditioning | `postpool_halfspace` tests the aggregation-reducible/post-pooling family; the primary has a same-pooled-value/different-output witness |
| [Value-aware Approximate Attention](https://aclanthology.org/2021.emnlp-main.753/) | Attention approximation objectives that include value vectors | It does not modify pairwise value direction or impose a ranking-readout constraint |
| [Max-Margin Token Selection](https://proceedings.neurips.cc/paper_files/paper/2023/hash/970f59b22f4c72aec75174aae63c7459-Abstract-Conference.html) | Theory of softmax attention optimization selecting locally optimal tokens | It characterizes learned attention weights; C39 constrains value vectors after selection |
| [BeliefFormer](https://openreview.net/forum?id=Ard2QzPAUK) | Orthogonal projection of an already aggregated attention value and tangent-style residual | C39 projects each event before aggregation onto a candidate-score halfspace; `postpool_halfspace` is the direct location ablation |
| C15 candidate-conditioned value write | Proves linear/bilinear pair values reduce to post-pool conditioning and unrestricted nonlinear values are generic edge messages | C39 supplies the previously missing fixed structural law and same-aggregate witness; it must beat post-pool and raw controls |
| C28 margin-local comparator | Candidate-relative readout but free direction/scale | C39 removes the direction gauge by a closed-form nonnegative local readout constraint |
| C35--C37 candidate transport | Relative admission and conservative candidate query geometry | C39 retains no tangent/barycentric transport; it changes event values at `V/W_O` and must beat the global-only control |

The `ray_only` control is especially binding. Under the immediate linear
candidate readout, both primary and ray-only have the same nonnegative scalar
contribution. Only the primary preserves score-neutral value components for
the downstream shared FFN. If they tie, C39 reduces to a scalar evidence boost
and fails the architecture-innovation test even if both beat the base.
