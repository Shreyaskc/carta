"""Synthetic/source checks for explicit manuscript contracts; no model runtime."""
import ast
import inspect
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import TrainingConfig
import gater_core
import probe


def extracted(path, name, namespace):
    node = next(node for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.FunctionDef) and node.name == name)
    node.returns = None
    for argument in node.args.args:
        argument.annotation = None
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class PaperAlignmentTests(unittest.TestCase):
    def test_cell_groups_continuous_but_masks_one_hot_elements(self):
        random = SimpleNamespace(
            split=lambda key, num=2: tuple(range(key * 10 + 1, key * 10 + 1 + num)),
            bernoulli=lambda key, p, shape: np.random.default_rng(key).random(shape) < p)
        factory = extracted(ROOT / 'augment.py', 'make_augmenter',
                            {'jax': SimpleNamespace(random=random), 'jnp': np, 'Dict': dict})
        batch = {'continuous': np.ones((32, 4)), 'one_hot': np.ones((32, 6)),
                 'embedding': np.ones((32, 2), dtype=int)}
        views = factory('cell', .5, [3, 3])(batch, 0)
        for view in views.values():
            self.assertTrue(np.all(view['continuous'] == view['continuous'][:, :1]))
            self.assertEqual(set(np.unique(view['continuous'])), {0., 1.})
            self.assertTrue(np.any(view['one_hot'] != view['one_hot'][:, :1]))
            np.testing.assert_array_equal(view['embedding'], batch['embedding'])

    def test_shared_categorical_head_retains_global_target_semantics(self):
        def ce(logits, labels):
            shifted = logits - logits.max(axis=1, keepdims=True)
            return np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(logits)), labels]
        state = SimpleNamespace(model_config=SimpleNamespace(one_hot_output_dim=4, embedding_cardinalities=[]),
            apply_fn=lambda *args, **kwargs: ((np.zeros((1, 1)), np.array([[2., -1., .3, .2]]), []),
                                             np.zeros((1, 2)), np.zeros((1, 2))))
        loss = extracted(ROOT / 'src/training.py', 'vae_loss_fn',
            {'jnp': np, 'optax': SimpleNamespace(l2_loss=lambda a,b: .5*(a-b)**2,
            softmax_cross_entropy_with_integer_labels=ce), 'state': state, 'params': {},
            'training': False, 'kl_weight': 0.})
        values = []
        for row in ([1., 0., 1., 0.], [1., 0., 0., 1.]):
            values.append(loss({'continuous': np.zeros((1,1)), 'one_hot': np.array([row]),
                                'embedding': np.empty((1,0))}, None, None)[0])
        self.assertEqual(*values)

    def test_default_recipe_and_probe_count(self):
        self.assertEqual(inspect.signature(gater_core.train_and_encode).parameters['excision_mode'].default, 'feature')
        self.assertEqual(TrainingConfig().epochs, 300)
        for function in (probe.evaluate, probe.label_efficiency):
            self.assertEqual(inspect.signature(function).parameters['seeds'].default, (0,1,2,3,4))

    def test_beta_sweep_rejects_unverified_merge_before_training(self):
        import pandas as pd
        import run_beta_sweep as sweep
        from datasets import DatasetSpec
        frame = pd.DataFrame({'amount': [1., 2.], 'label': [0, 1]})
        spec = DatasetSpec('synthetic', ['amount'], [], 'label')
        self.assertEqual(sweep.PROBE_SEEDS, (0,1,2,3,4))
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'beta_sweep_synthetic.json').write_text(json.dumps(
                {'dataset': 'synthetic', 'seed': 0, 'configs': {'cyclical': {}}}))
            with patch.object(sweep, 'RESULTS', directory), patch.object(sweep, 'DS', 'synthetic'), \
                 patch.object(sweep.datasets, 'load', return_value=(frame, spec)), \
                 patch.object(sweep, 'train_and_encode') as encode, self.assertRaises(ValueError):
                sweep.main()
            encode.assert_not_called()

    def test_checkpoint_selection_uses_reconstruction_in_both_loops(self):
        for path, target in ((ROOT / 'gater_core.py', 'vloss'), (ROOT / 'src/training.py', 'avg_val_loss')):
            tree = ast.parse(path.read_text())
            assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == target for t in node.targets))
            value = eval(compile(ast.Expression(assignment.value), str(path), 'eval'),
                         {'jax': SimpleNamespace(device_get=lambda v: v),
                          'vm': {'total': 999., 'reconstruction': 3.},
                          'val_metrics': {'total': 999., 'reconstruction': 3.}})
            self.assertEqual(value, 3.)


if __name__ == '__main__':
    unittest.main()
