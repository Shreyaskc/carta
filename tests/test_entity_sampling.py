"""Exercise entity-generator sample selection without plotting/UMAP runtimes."""
import ast
from pathlib import Path
import unittest
import numpy as np


class EntitySamplingTests(unittest.TestCase):
    def test_valid_small_populations_are_sampled_without_replacement(self):
        root = Path(__file__).resolve().parents[1]
        for script, controls in (("gen_tabformer_entity.py", {"N_FIT", "fit_idx", "N_VIZ", "viz_idx"}),
                                 ("gen_umap_entity.py", {"N_VIZ", "idx"})):
            tree = ast.parse((root / script).read_text())
            selected = [node for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id in controls for target in node.targets)]
            for size in (8, 100, 100_000):
                namespace = {"np": np, "rng": np.random.default_rng(42), "emb": np.zeros((size, 16))}
                exec(compile(ast.Module(body=selected, type_ignores=[]), script, "exec"), namespace)
                for name in controls - {"N_FIT", "N_VIZ"}:
                    indices = namespace[name]
                    self.assertGreater(len(indices), 0)
                    self.assertLessEqual(len(indices), size)
                    self.assertEqual(len(np.unique(indices)), len(indices))
                    self.assertLess(indices.max(), size)


if __name__ == "__main__":
    unittest.main()
