# Code and experiment map

All commands below refer to files in the repository root. Training and generator commands are interface examples, not validated reproduction claims. Complete the prerequisites in [reproducibility](reproducibility.md) before running them.

| Purpose | Implementation / entry point | Paper relationship |
| --- | --- | --- |
| Feature types and public loaders | [datasets.py](../datasets.py), [adapter.py](../adapter.py) | Heterogeneous input representation |
| Preprocessing | [data_processing.py](../src/data/data_processing.py), [temporal_preprocessor.py](../src/data/temporal_preprocessor.py) | Numeric transformation and categorical vocabularies |
| Cross-attention VAE | [vae.py](../src/models/vae.py), [encoders.py](../encoders.py) | Encoder and architecture variants |
| Objective / augmentation | [training.py](../src/training.py), [augment.py](../augment.py) | Reconstruction, KL, InfoNCE and excision |
| Configuration | [config.py](../src/config.py), [ablations.py](../ablations.py) | Defaults and named comparisons |
| Encoder orchestration | [gater_core.py](../gater_core.py), [gater_runner.py](../gater_runner.py) | Frozen embeddings |
| Baseline representations | [baselines.py](../baselines.py), [run_baselines.py](../run_baselines.py) | Raw, PCA and frequency encoding, plus supplied embeddings |
| Artifact identity | [experiment_io.py](../experiment_io.py) | Ordered row/protocol checks before scoring |
| Frozen probes | [probe.py](../probe.py) | Linear/GBDT metrics and label efficiency |
| Component comparisons | [run_ablations.py](../run_ablations.py) | Ablation results |
| Supervised comparison | [catboost_baseline.py](../catboost_baseline.py) | Native-categorical supervised classifier |
| Encoder-seed protocol | [run_reference_seeds.py](../run_reference_seeds.py) | Separates training seeds from probe seeds |
| KL sweep | [run_beta_sweep.py](../run_beta_sweep.py) | Feature-excision KL/beta comparisons |
| Reference convenience runner | [run_c2_reference.py](../run_c2_reference.py) | One trained embedding per dataset, followed by several probe seeds; not a multi-training-seed result |
| Representation families | [run_representation_comparison.py](../run_representation_comparison.py) | PCA/UMAP/AE/SAE/VAE/SCARF-style/CARTA comparisons |
| Temporal checks | [run_splits.py](../run_splits.py) | IEEE-CIS temporal protocol; does not test merchant-disjoint TabFormer |
| Clustering stability | [cluster_stability.py](../cluster_stability.py) | Repeated K-means adjusted Rand index |
| Label-efficiency error bars | [compute_labeleff_std.py](../compute_labeleff_std.py) | Repeated probe evaluation; requires embeddings |
| Paper numeric output | [gen_macros.py](../gen_macros.py) | Saved JSON to LaTeX macros, plus a manual-value block |
| Paper figures | [gen_figures.py](../gen_figures.py), [gen_umap_entity.py](../gen_umap_entity.py), [gen_tabformer_entity.py](../gen_tabformer_entity.py) | Result plots and exploratory entity/UMAP plots |
| Supported starting example | [synthetic_probe.py](../examples/synthetic_probe.py) | CPU demonstration, not a paper experiment |
| Regression checks | [tests/](../tests/) | Synthetic protocol, CLI, configuration, numerical and artifact checks |

Planned encoder interface:

```sh
python gater_runner.py --dataset ieee_cis --epochs 300 --excision-mode feature --out results/new-runs/gater_ieee_cis.npy
python run_baselines.py --dataset ieee_cis --seeds 0,1,2,3,4 --embeddings results/new-runs/gater_ieee_cis.npy
```

Do not interpret that pair as five independent encoder runs. Use the separate training-seed runner for that protocol, after reviewing its overrides and output destination. Explicitly set `METHODS` for the representation-family runner; do not silently select a default comparison list.
