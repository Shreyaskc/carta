import logging

import numpy as np
import pandas as pd
import powerlaw
from scipy import stats
from scipy.special import inv_boxcox


class SmartPowerlawTransformer:
    """
    A transformer that automatically detects and transforms numerical columns
    with heavy-tailed (Pareto-like) distributions.

    The transformation is a two-step process: log(x+epsilon) followed by Box-Cox.
    """
    def __init__(self, sample_size=100_000, significance_level=0.05, continuous_cols=None):
        """
        Initializes the transformer.

        Args:
            sample_size (int): The number of rows to sample for the power-law fit.
            significance_level (float): The p-value threshold for the
                                        power-law vs. exponential comparison.
        """
        self.sample_size = sample_size
        self.significance_level = significance_level
        self.lambdas_ = {}
        self.shifts_ = {}
        self.transformed_columns_ = []
        self.continuous_cols = continuous_cols

    def fit(self, X: pd.DataFrame, y=None):
        """
        Analyzes the DataFrame to find power-law-like columns and fits the
        necessary transformation parameters.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            The fitted instance of self.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        self.lambdas_ = {}
        self.shifts_ = {}
        self.transformed_columns_ = []
        if self.continuous_cols:
            numeric_cols = self.continuous_cols
        else:
            numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

        for col in numeric_cols:
            data = X[col].dropna()
            if data.empty:
                continue

            # Ensure data is positive for power-law and Box-Cox
            min_val = data.min()
            shift = 0
            if min_val <= 0:
                shift = abs(min_val) + 1e-6  # Add a small epsilon
            self.shifts_[col] = shift
            data_for_fit = data + shift

            # Sample data for fitting to ensure scalability
            if len(data_for_fit) > self.sample_size:
                data_for_fit = data_for_fit.sample(self.sample_size, random_state=42)

            # Fit powerlaw and compare to exponential distribution
            try:
                fit = powerlaw.Fit(data_for_fit, verbose=False)
                # R is the log-likelihood ratio, p is the significance
                R, p_value = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
            except Exception as e:
                logging.warning(f"Could not perform power-law fit on column '{col}': {e}. Skipping.")
                continue

            # If R > 0, power_law is a better fit. If p < threshold, the result is statistically significant.
            if R > 0 and p_value < self.significance_level:
                logging.info(
                    f"Column '{col}' detected as power-law like (R={R:.2f}, p={p_value:.3f}). "
                    "Marking for transformation."
                )
                self.transformed_columns_.append(col)

                # Fit Box-Cox lambda on the log-transformed data
                log_transformed_data = np.log1p(data_for_fit)
                _, lmbda = stats.boxcox(log_transformed_data)
                self.lambdas_[col] = lmbda
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the two-step transformation to the identified columns."""
        X_transformed = X.copy()
        for col in self.transformed_columns_:
            shift = self.shifts_.get(col, 0)
            lmbda = self.lambdas_[col]

            # Apply log(x+1) then Box-Cox
            shifted_data = X_transformed[col] + shift
            if not np.isfinite(shifted_data).all() or (shifted_data <= 0).any():
                raise ValueError(f"Column {col!r} contains missing/nonfinite values or values outside the fitted positive transform domain")
            log_data = np.log1p(shifted_data)
            X_transformed[col] = stats.boxcox(log_data, lmbda=lmbda)
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """A convenience method to fit and then transform the data."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed: pd.DataFrame) -> pd.DataFrame:
        """Reverses the transformation to restore the data to its original scale."""
        X_original = X_transformed.copy()
        for col in self.transformed_columns_:
            if col not in self.lambdas_: continue

            shift = self.shifts_.get(col, 0)
            lmbda = self.lambdas_[col]

            # Apply inverse Box-Cox then inverse log(x+1) (which is exp(y)-1)
            inv_boxcox_data = inv_boxcox(X_original[col], lmbda)
            inv_log_data = np.expm1(inv_boxcox_data)

            # Subtract the original shift
            X_original[col] = inv_log_data - shift
        return X_original
