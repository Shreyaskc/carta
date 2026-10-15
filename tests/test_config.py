import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ModelConfig, TrainingConfig, apply_overrides, validate_configs
from test_training_boundaries import helper
import numpy as np


class ConfigTests(unittest.TestCase):
    def test_unknown_override_rejected_and_effective_values_retained(self):
        config = TrainingConfig()
        with self.assertRaises(ValueError):
            apply_overrides(config, {"batch_szie": 8})
        apply_overrides(config, {"batch_size": 8, "random_state": 7})
        validate_configs(ModelConfig(), config)
        self.assertEqual((config.batch_size, config.random_state), (8, 7))

    def test_invalid_domains_rejected(self):
        for overrides in ({"batch_size": 0}, {"epochs": -1}, {"contrastive_temperature": 0},
                          {"augmentation_mask_prob": 1.1}, {"warmup_epochs": -1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_configs(ModelConfig(), apply_overrides(TrainingConfig(), overrides))
        with self.assertRaises(ValueError):
            validate_configs(ModelConfig(feature_embedding_dim=63), TrainingConfig())

    def test_zero_and_tiny_latents_normalize_finitely(self):
        normalize = helper("normalize_latents", {"jnp": np})
        result = normalize(np.array([[0., 0.], [1e-30, 0.], [3., 4.]]))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result[-1], [0.6, 0.8])


if __name__ == "__main__":
    unittest.main()
