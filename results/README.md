# Saved experiment records

These twelve historical JSON records support inspection of the reported experiments. They have not been independently reproduced in this environment. The records contain seed/error-bar and run-identity differences; see the reproducibility guide. These records remain unchanged and must be reconciled before table regeneration.

| Record group | Meaning |
| --- | --- |
| `baselines_*` | Raw/PCA/frequency/CARTA frozen probes and label efficiency |
| `ablations_*` | Component and architecture comparisons; `full` is cell mode |
| `catboost_*` | Fully supervised native-categorical comparison |
| `representation_comparison_ieee_cis` | Representation families, probe metrics, silhouette and timing |
| `refseeds_*_ieee_cis` | Training-seed records for cell/feature/SCARF-style variants |
| `splits_ieee_cis_temporal` | IEEE-CIS temporal check |
| `labeleff_std` | Probe-seed error bars |

Nine metric values are unavailable and represented as `null`. Never treat them as zero or infer digits from another run:

| File | Unavailable metric |
| --- | --- |
| `baselines_tabformer.json` | `full_label.freq.gbdt.auc_std` |
| `refseeds_cell_ieee_cis.json` | `full_label.gater16.linear.auprc` |
| `refseeds_cell_ieee_cis.json` | `full_label.gater16.linear.auprc_std` |
| `refseeds_cell_ieee_cis.json` | `full_label.gater16.linear.auc` |
| `refseeds_cell_ieee_cis.json` | `full_label.gater16.linear.auc_std` |
| `refseeds_cell_ieee_cis.json` | `full_label.gater16.gbdt.auprc` |
| `refseeds_cell_ieee_cis.json` | `full_label.gater16.gbdt.auprc_std` |
| `ablations_tabformer.json` | `results.full.full_label.linear.auprc_std` |
| `catboost_ieee_cis.json` | `label_efficiency.catboost.1.00.auc` |

Other null values, such as an unavailable training time, retain their original missing-value meaning. No missing value has been replaced by an estimated result. Macro/figure generation is unvalidated and does not constitute rerunning experiments. See [reproducibility](../docs/reproducibility.md).
