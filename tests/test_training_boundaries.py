"""Isolated training helpers with NumPy substitutions; no JAX training claimed."""
import ast
import logging
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import TrainingConfig


def helper(name, namespace):
    path = Path(__file__).resolve().parents[1] / "src/training.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class TrainingBoundaryTests(unittest.TestCase):
    def test_legacy_tiny_loop_performs_one_update(self):
        class Progress:
            def __init__(self, iterable, **kwargs):
                self.iterable = iterable
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def __iter__(self):
                return iter(self.iterable)
            def set_postfix(self, **kwargs):
                pass
        calls = []
        def step(state, batch, total):
            calls.append(total)
            return state, {"total": 1., "kl_weight": 0.1}
        runtime = SimpleNamespace(device_get=lambda value: value,
                                  tree_util=SimpleNamespace(tree_map=lambda function, batch: {k: function(v) for k, v in batch.items()}))
        loop = helper("run_training_loop", {"jax": runtime, "jnp": np, "train_step": step,
                                           "tqdm": Progress, "logging": logging,
                                           "run_evaluation": lambda *args: {"total": 1., "reconstruction": 0., "kl": 0.}})
        state = SimpleNamespace(training_config=TrainingConfig(epochs=1, batch_size=16))
        loop(state, iter([{"continuous": np.ones((3, 1))}]), {"continuous": np.ones((1, 1))}, 3)
        self.assertEqual(calls, [1])

    def test_scheduler_receives_total_length_including_warmup(self):
        captured = []
        def schedule(**kwargs):
            self.assertGreater(kwargs["decay_steps"], kwargs["warmup_steps"])
            captured.append(kwargs)
            return None
        optimizer = SimpleNamespace(init=lambda params: None)
        optax = SimpleNamespace(warmup_cosine_decay_schedule=schedule,
                                chain=lambda *args: optimizer, clip_by_global_norm=lambda value: None,
                                adam=lambda *args, **kwargs: optimizer)
        jax = SimpleNamespace(random=SimpleNamespace(split=lambda key: (1, 2)))
        initialize = helper("initialize_train_state", {"jax": jax, "jnp": np, "optax": optax,
                                                       "TrainState": lambda **kwargs: SimpleNamespace(**kwargs)})
        model = SimpleNamespace(init=lambda *args, **kwargs: {"params": {}}, apply=lambda *args: None)
        from src.config import ModelConfig
        for epochs in (11, 15, 20, 21):
            initialize(lambda config: model, ModelConfig(), TrainingConfig(epochs=epochs, batch_size=2), 0, 20)
            self.assertEqual(captured[-1]["decay_steps"], epochs * 10)

    def test_legacy_field_keys_are_stable_across_hash_seeds(self):
        path = Path(__file__).resolve().parents[1] / "src/training.py"
        code = ("import ast,hashlib; from pathlib import Path; "
                f"tree=ast.parse(Path({str(path)!r}).read_text()); "
                "f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stable_path_id'); "
                "exec(compile(ast.Module(body=[f],type_ignores=[]),'key','exec')); "
                "print(stable_path_id(('continuous',)))")
        values = []
        for seed in ("1", "2"):
            values.append(subprocess.check_output([sys.executable, "-c", code],
                                                 env=dict(os.environ, PYTHONHASHSEED=seed), text=True))
        self.assertEqual(*values)

    def test_short_kl_cycles_are_finite_and_zero_cycles_rejected(self):
        schedule = helper("calculate_annealed_kl_weight", {"jnp": np})
        for total in (1, 2, 9):
            self.assertTrue(np.isfinite(schedule(TrainingConfig(), 0, total)))
        with self.assertRaises(ValueError):
            schedule(TrainingConfig(kl_anneal_cycles=0), 0, 100)

    def test_tiny_dataset_yields_and_empty_dataset_rejected(self):
        generator = helper("data_generator", {"np": np})
        data = {"continuous": np.arange(3).reshape(-1, 1)}
        batch = next(generator(data, 16, 0))
        self.assertEqual(set(batch["continuous"][:, 0]), {0, 1, 2})
        with self.assertRaises(ValueError):
            next(generator({"continuous": np.empty((0, 1))}, 16, 0))

    def test_validation_includes_tail_with_row_weighting(self):
        visited = []
        def evaluate(state, batch):
            visited.extend(batch["continuous"][:, 0].tolist())
            return {"total": np.mean(batch["continuous"])}
        runtime = SimpleNamespace(tree_util=SimpleNamespace(tree_map=lambda function, batch: {k: function(v) for k, v in batch.items()}))
        evaluation = helper("run_evaluation", {"jax": runtime, "jnp": np, "eval_step": evaluate})
        data = {"continuous": np.array([0., 0., 0., 0., 10.]).reshape(-1, 1)}
        result = evaluation(None, data, 4)
        self.assertEqual(len(visited), 5)
        self.assertEqual(result["total"], 2.)


if __name__ == "__main__":
    unittest.main()
