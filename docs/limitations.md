# Scientific and implementation limitations

The demonstrated scope is fraud probing and exploratory clustering. Forecasting, drift monitoring, broad task transfer and cross-attention scaling are not established by the available results. Token counts of 2 and 7 do not exercise a large-input scaling regime.

The main protocol is transductive: pooled feature distributions influence preprocessing and encoding even without test labels. Random transaction splits can share merchants across partitions. The IEEE-CIS temporal check does not resolve TabFormer merchant leakage, and no train-only merchant-disjoint TabFormer evaluation is verified here.

Fully supervised CatBoost is stronger than the reported CARTA tree probe on TabFormer. The weak TabFormer linear-probe result and the substantially different fixed-beta ablation require matched configuration/seed analysis. CARTA should not be described as uniformly superior to supervised boosting or label-free alternatives.

Important implementation checks remain:

- End-to-end JAX/Flax training, benchmark evaluation and generators are unvalidated. The legacy low-level training loop is incomplete and is not a supported paper runner.
- One-hot reconstruction currently uses a global argmax over concatenated categorical blocks. A per-field likelihood would change the objective and require fresh experiments; it has not been silently substituted.
- The SCARF-style augmenter leaves identifier embeddings unchanged, permits same-row replacement and reuses a PRNG key for masking/replacement. Changing these details would require fresh result provenance.
- `no_contrastive` also disables augmentation; the reference ablation is cell mode while feature excision is separate.
- Some convenience scripts choose recipes/defaults differently. No single canonical configuration is claimed to match every historical table.
- Dataset bytes, embeddings and checkpoints are absent. Saved result values are historical records, with nine unavailable fields.

Final categorical-loss semantics, historical table/seed provenance, author metadata and production requirements remain pending. No manuscript is bundled.
