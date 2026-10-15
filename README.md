# CARTA

CARTA learns compact transaction representations from mixed numeric and categorical features without fraud labels. A frozen embedding can then be evaluated with supervised fraud probes and exploratory clustering.

This is the companion implementation for **CARTA: Self-Supervised Reusable Transaction Embeddings via Heterogeneous Tokenization and Cross-Attention**, ICAIF 2026. Final author and publication metadata will be added before release; no DOI or software license is currently supplied.

The implemented encoder projects numeric and low-cardinality inputs into shared tokens, embeds high-cardinality identifiers, and uses learned queries with cross-attention to produce a 16-dimensional latent mean. Training combines reconstruction, KL regularization and contrastive learning. See [architecture](docs/architecture.md) for the exact data flow and important implementation details.

## Start here

Run from the repository root. The supported CPU example uses synthetic data and does not download datasets or train the CARTA encoder:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python examples/synthetic_probe.py
python -m unittest discover -s tests -v
```

The example demonstrates feature roles, baseline representations and a fraud probe. Its numbers are not scientific benchmark results. Full encoder training and paper generators have not been validated end to end in this environment.

## Find your way around

- [Architecture and configuration](docs/architecture.md): tokenization, latent encoder, loss and settings.
- [Code and experiment map](docs/code-map.md): modules and their relationship to the paper experiments.
- [Data and reproducibility](docs/reproducibility.md): inputs, saved records, training and seed protocols.
- [Scientific and implementation limitations](docs/limitations.md): what the evidence supports and what remains unvalidated.
- [Saved experiment records](results/README.md): twelve historical records, including explicitly unavailable values.

The main entry points are [gater_runner.py](gater_runner.py) for encoder training, [run_baselines.py](run_baselines.py) for representation probes, and [run_ablations.py](run_ablations.py) for component comparisons. Historical `gater` identifiers in code and result keys refer to CARTA.

## Status

The CPU example and targeted synthetic tests are supported. Training, figure generation, full model serialization and benchmark metrics remain unverified. Raw datasets, learned embeddings, checkpoints and the manuscript are not included. The implementation has known method ambiguities and historical result discrepancies; see the limitations and reproducibility guides.

See [citation guidance](docs/citation.md) before citing or releasing this package.
