# Architecture and configuration

## From a transaction to an embedding

1. [datasets.py](../datasets.py) returns a dataframe and `DatasetSpec`, identifying numeric, categorical and label columns. [adapter.py](../adapter.py) classifies categoricals by observed cardinality; its threshold defaults to 50.
2. [SmartPowerlawTransformer](../src/data/data_processing.py) optionally transforms continuous inputs. [TabularPreprocessor](../src/data/temporal_preprocessor.py) fits numeric quantile transforms, one-hot categories and identifier vocabularies. Unknown identifiers map to index 0.
3. [MultiQueryPerceiverEncoder](../src/models/vae.py) creates one token for the entire continuous block, one for the entire one-hot block, and one per high-cardinality field. All tokens have width 64 by default and receive learned positional embeddings. More columns do not necessarily increase token count.
4. Eight learned latent queries attend to the input tokens. Four blocks apply cross-attention, query self-attention and a feed-forward network with residual normalization. Averaging queries gives a context vector; two projections produce a 16-dimensional mean and log variance.
5. During training, a sampled latent goes through [Decoder](../src/models/vae.py): shared hidden layers followed by continuous, one-hot and identifier reconstruction heads. The frozen representation uses the latent **mean**, with the decoder discarded for probing.

[encoders.py](../encoders.py) supplies the experiment variants: the Perceiver, a cross-attention/GRU-gated context encoder and an MLP encoder. Nonvariational settings support reconstruction-only AE/SAE comparisons in the representation runner; these are different training objectives, not interchangeable CARTA configurations.

## Augmentation and objective

[augment.py](../augment.py) produces two independently keyed views:

| Mode | Numeric / one-hot behavior | Identifier behavior |
| --- | --- | --- |
| `cell` | Whole continuous block; elementwise one-hot Bernoulli masking | IDs unchanged |
| `feature` | Whole continuous block and per-category one-hot block masking | Dropped IDs become 0 |
| `scarf` | Marginal row-value replacement | IDs unchanged |

The SCARF-style variant can select the same row as replacement; it does not guarantee a different row. It is an augmentation variant within this architecture, not evidence that every external SCARF implementation has been matched.

[training.py](../src/training.py) computes reconstruction plus weighted KL and, when enabled, symmetric InfoNCE on normalized latent means. The two augmented VAE losses are averaged. The implemented one-hot reconstruction uses a global argmax over the concatenated one-hot vector; it is not a separate likelihood for every categorical field. See [limitations](limitations.md).

With cyclical annealing enabled, beta ramps to its maximum in the first half of each cycle and stays there in the second half. Disabling annealing returns fixed `kl_weight_max`; it does **not** remove KL.

## Settings and experiments

[src/config.py](../src/config.py) defines model and training settings. Defaults include token width 64, eight queries, four layers/heads, latent width 16, dropout 0.1, excision probability 0.2, contrastive temperature 0.07 and contrastive weight 0.1. Training and wrapper defaults use at most 300 epochs, with reconstruction-based validation checkpoint selection and patience 50. The main API/CLI default to feature excision; the Table 3 ablation reference explicitly retains cell mode.

[gater_core.py](../gater_core.py) applies dataset-specific feature roles and model/training overrides. [ablations.py](../ablations.py) defines named variants. Its `full` reference uses **cell** mode; `excision_feature` is a separate row. The `no_contrastive` variant also disables augmentation, so it is a compound ablation.

The exact categorical objective and compound ablation limitations are described above and in the limitations guide.

The standard wrapper fits preprocessing on pooled unlabeled features before splitting encoder train/validation data. The later supervised fraud split is separate. This is a transductive protocol, not train-only inductive generalization.

The current training wrapper requires at least one continuous input field. Configuration validation rejects invalid dimensions, probabilities and temperature settings before fitting data. The MLP variant preserves field order through concatenation and does not add the learned positional embeddings used by the attention variants; architecture comparisons are not strict parameter-count matches.
