import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer


class TabularPreprocessor:

    def __init__(
        self,
        continuous_cols: List[str],
        one_hot_cols: List[str],
        embedding_cols: List[str],
        random_state: int = 42
    ):
        self.random_state = random_state
        self.continuous_cols = continuous_cols
        self.one_hot_cols = one_hot_cols
        self.embedding_cols = embedding_cols
        self.is_fitted = False
        self.transformers: Dict[str, Any] = {}
        self.embedding_maps: Dict[str, Dict[str, int]] = {}
        self.metadata: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame):
        """
        Fits all transformers on a reference dataframe.

        This method should only be called once on the initial "normal" dataset.

        Args:
            df (pd.DataFrame): The reference training data to fit the preprocessor on.
        """
        logging.info("Fitting TabularPreprocessor on reference data...")
        if self.continuous_cols:
            n_quantiles = max(min(len(df) // 10, 1000), 10)
            self.transformers['continuous'] = QuantileTransformer(
                output_distribution='normal', n_quantiles=n_quantiles, random_state=getattr(self, 'random_state', 42)
            )
            self.transformers['continuous'].fit(df[self.continuous_cols])

        if self.one_hot_cols:
            self.transformers['one_hot'] = OneHotEncoder(
                handle_unknown='ignore', sparse_output=False
            )
            self.transformers['one_hot'].fit(df[self.one_hot_cols])

        if self.embedding_cols:
            for col in self.embedding_cols:
                categories = ['<unknown>'] + [category for category in df[col].astype('category').cat.categories.tolist() if category != '<unknown>']
                self.embedding_maps[col] = {cat: i for i, cat in enumerate(categories)}

        self._generate_metadata(df)
        self.is_fitted = True
        logging.info("Preprocessor fitting complete. Ready for temporal transformation.")
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Transforms a dataframe using the pre-fitted transformers.

        Args:
            df (pd.DataFrame): The data from any time period (reference or subsequent) to transform.

        Returns:
            A dictionary of processed numpy arrays ready for the model.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on reference data before transforming.")

        output = {}
        if self.continuous_cols:
            output['continuous'] = self.transformers['continuous'].transform(df[self.continuous_cols]).astype(np.float32)
        else:
            output['continuous'] = np.zeros((len(df), 0), dtype=np.float32)

        if self.one_hot_cols:
            output['one_hot'] = self.transformers['one_hot'].transform(df[self.one_hot_cols]).astype(np.float32)
        else:
            output['one_hot'] = np.zeros((len(df), 0), dtype=np.float32)

        if self.embedding_cols:
            embedding_data = np.zeros((len(df), len(self.embedding_cols)), dtype=np.int32)
            for i, col in enumerate(self.embedding_cols):
                map_dict = self.embedding_maps[col]
                embedding_data[:, i] = df[col].map(lambda x: map_dict.get(x, 0)).values
            output['embedding'] = embedding_data
        else:
            output['embedding'] = np.zeros((len(df), 0), dtype=np.int32)

        return output

    def inverse_transform(self, reconstructions: Tuple, index: pd.Index) -> pd.DataFrame:
        """
        Converts reconstructed data back into a human-readable DataFrame.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse transforming.")

        recon_cont, recon_one_hot_logits, recon_embed_logits = reconstructions
        df_recon = pd.DataFrame(index=index)

        # 1. Inverse transform continuous features
        if self.continuous_cols:
            continuous_scaler = self.transformers['continuous']
            recon_cont_np = np.asarray(recon_cont)

            recon_cont_clipped = np.clip(recon_cont_np, -5, 5)

            df_recon[self.continuous_cols] = continuous_scaler.inverse_transform(recon_cont_clipped)

        # 2. Inverse transform one-hot encoded features from logits
        if self.one_hot_cols:
            one_hot_encoder = self.transformers['one_hot']
            start_idx = 0
            for i, cats in enumerate(one_hot_encoder.categories_):
                num_cats = len(cats)
                logits_subset = np.asarray(recon_one_hot_logits[:, start_idx : start_idx + num_cats])
                pred_indices = np.argmax(logits_subset, axis=1)
                original_col_name = self.one_hot_cols[i]
                df_recon[original_col_name] = cats[pred_indices]
                start_idx += num_cats

        # 3. Inverse transform embedding features from logits
        if self.embedding_cols:
            for i, col in enumerate(self.embedding_cols):
                logits = np.asarray(recon_embed_logits[i])
                pred_indices = np.argmax(logits, axis=1)
                inv_map = {v: k for k, v in self.embedding_maps[col].items()}
                df_recon[col] = [inv_map.get(idx, '<error>') for idx in pred_indices]

        all_cols = self.continuous_cols + self.one_hot_cols + self.embedding_cols
        return df_recon[all_cols]


    def _generate_metadata(self, df: pd.DataFrame):
        """Generates metadata required by the model from the fitted transformers."""
        self.metadata['embedding_cardinalities'] = [len(self.embedding_maps[col]) for col in self.embedding_cols]
        if self.one_hot_cols:
            cats = self.transformers['one_hot'].categories_
            self.metadata['one_hot_output_dim'] = sum(len(c) for c in cats)
        else:
            self.metadata['one_hot_output_dim'] = 0
        self.metadata['continuous_output_dim'] = len(self.continuous_cols)

    def save(self, filepath: Path):
        """Saves the fitted preprocessor to a file."""
        joblib.dump(self, filepath)
        logging.info(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath: Path) -> 'TabularPreprocessor':
        """Loads a preprocessor from a file."""
        logging.info(f"Loading preprocessor from {filepath}")
        return joblib.load(filepath)
