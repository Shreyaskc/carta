# Data and reproducibility

## Supported local checks

From the repository root, install [requirements-cpu.txt](../requirements-cpu.txt), then run:

```sh
python examples/synthetic_probe.py
python -m unittest discover -s tests -v
```

The validated CPU environment used Python 3.14.6, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1 and joblib 1.6.0 on this Mac. Other dependency combinations are untested.

These use synthetic inputs. They check feature-role handling, baseline dimensions, split isolation, artifact identity, configuration, serialization and training boundaries. Scalar/batching training tests use explicit NumPy/runtime substitutions; they do not validate JAX execution or gradients. They do not reproduce fraud benchmarks or validate JAX model training.

[requirements.txt](../requirements.txt) lists the broader experiment dependencies, including JAX, Flax, Optax, matplotlib, UMAP, CatBoost and powerlaw. That complete environment has not been validated here. Select a JAX CPU/CUDA build appropriate to the target machine; GPU installation and hardware compatibility are outside the validated quickstart. Dependencies are not a locked scientific environment.

## Inputs

Loaders now resolve `.data/` at the repository root:

- IEEE-CIS: `.data/ieee.parquet`. The loader requires the named raw-feature columns in `datasets.py` and prints its configured download URL when absent. Verify dataset identity and preprocessing before comparing metrics.
- TabFormer: `.data/tabformer/card_transaction.v1.csv` (or one of the parquet/alternative filenames accepted by the loader). Obtain the benchmark from [IBM/TabFormer](https://github.com/IBM/TabFormer). No dataset is included.
- Frozen embeddings: `results/gater_<dataset>.npy`, one row per loaded transaction in exactly the loader's row order. Row count alone cannot establish correct alignment. Sampling seed and categorical-collapse settings must match between encoding and probing.

TabFormer settings include `TABFORMER_SAMPLE` and `TABFORMER_MINFREQ`; explicitly reproduce them rather than assuming a default 2M-row sample. Check memory and compute requirements before starting a run. No measured runtime estimate for the current package is supplied.

## Saved records versus new experiments

[results/](../results/README.md) contains twelve historical experiment records. Nine values are unavailable and represented as `null`. They have not been independently reproduced. Do not use a null as zero or fill it from another configuration.

`python gen_macros.py` and `python gen_figures.py` consume saved records; neither reruns training. They have not been executed or verified in this package. Resolve unavailable fields and validate output before use. Macro output also includes manually entered EP*, Cluster* and FTT* values; those require separate source verification and are not guaranteed by the JSON files. Exploratory UMAP/entity generation requires embeddings which are not included.

Encoder runs are separate from regeneration. The reference protocol uses pooled unlabeled feature fitting followed by stratified 70/30 probe splits; probes fit on training rows and score the held-out rows. Record dataset/version, preprocessing fit population, excision mode, KL settings, encoder seeds, probe seeds, label-subsampling seeds and checkpoint selection for every result. Probe interfaces now default to five seeds (0–4), matching the manuscript count; exact seed IDs are an explicit implementation choice because the paper does not enumerate them. Historical reference records used three inner probe seeds; do not relabel those records.

Training-seed uncertainty requires independently trained embeddings. Repeated probes on one embedding do not measure training-seed uncertainty. Historical tables/configurations differ; do not combine them into matched ablations without configuration and seed checks. Never select a final recipe using test performance alone.

New experiment runners default to `results/new-runs/` to avoid overwriting shipped historical records. `GATER_RESULTS` can override their directory; supply a matching explicit `--embeddings`/`--out` path where supported. Saved-record generators continue to read the historical `results/` directory unless their interface provides an override. Entity generators instead default to `results/new-runs/`, matching new encoder outputs. Use consistent sample settings and row order when loading new embeddings.

## Artifact and population controls

Keep `gater_<dataset>.metadata.json` alongside `gater_<dataset>.npy`. New encoder outputs record dataset identity, ordered unlabeled feature fingerprint, array shape, a SHA-256 digest of the saved array, and resolved representation settings. Baseline/error-bar/split readers and entity generators verify row identity and array content before scoring. Reused CARTA representations in the comparison runner require 16 columns and check the resolved architecture, optimization, augmentation, seed, batch, cardinality and objective settings; the actual reused metadata is included with its scores.

Legacy embeddings without metadata require an explicit choice: `--allow-unverified-embeddings` for the baseline CLI, or `ALLOW_UNVERIFIED_EMBEDDINGS=1` for the comparison/error-bar/split/entity interfaces. Missing-sidecar opt-ins warn that row identity is unverified; checksum-free older sidecars also require this opt-in and warn that array content is unverified. Neither confers verified provenance. Preserve sidecars when moving files.

`TABFORMER_NROWS` selects the first positive number of source rows before optional fixed-seed `TABFORMER_SAMPLE` sampling. CSV and parquet obey the same head limit. Changing head/sample/category-collapse settings changes population identity. Input limits must be positive when supplied.

Override keys must be declared non-derived configuration fields; unknown keys fail rather than silently doing nothing. Seeds must be integers in `[0, 2**32)`. In the reference-seed runner, use `SEEDS`, `EPOCHS` and `BATCH` for those controls; do not also set `random_state`, `epochs` or `batch_size` in `TRAIN_OVERRIDES`. Existing baseline population and probe-seed settings must match before its CARTA row can be updated. Partial representation-comparison runs likewise reject incompatible protocol merges. Baseline/reference and comparison merge checks bind scores to ordered features and labels; older records without this evaluation fingerprint cannot be merged. Select a separate output directory instead.

## Behavior and historical metrics

The current implementation uses seeded quantile preprocessing, corrected effective batch/seed settings, complete sample-weighted validation, manuscript-aligned feature defaults, groupwise continuous masking in cell mode, reconstruction-based checkpoint selection, and the total schedule length expected by [Optax's warmup-cosine API](https://optax.readthedocs.io/en/latest/api/generated/optax.schedules.warmup_cosine_decay_schedule.html). These changes can affect learned embeddings and checkpoint selection. These include behavior changes relative to earlier experiment implementations. Historical records have not been recomputed, and new runs must not be presented as reproducing them without verification.

The categorical reconstruction objective has not been changed: it still uses a global argmax across concatenated one-hot blocks. A per-field objective would be a scientific change, requiring an explicit method decision and fresh experiments. Numerical guards, tiny-batch handling and malformed-input rejection have synthetic coverage, not benchmark validation.
