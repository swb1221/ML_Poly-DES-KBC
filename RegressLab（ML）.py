"""
Machine Learning Regression Analysis Framework
==============================================

This module provides a comprehensive pipeline for regression analysis,
particularly suited for small-sample, multi-feature datasets with a
continuous target variable (e.g., extraction yield prediction). It includes:

- Data loading and preprocessing (missing values, outliers, data augmentation)
- Model training and evaluation (linear, tree-based, ensemble, SVM, neural networks)
- Rich visualizations (fit curves, R² comparison, scatter plots, residual plots, correlation heatmap)
- Model interpretation (SHAP, PDP)
- Feature interaction analysis (3D surface plots)
- Optimal parameter search based on the best model

Dependencies:
    numpy, pandas, matplotlib, scikit-learn, xgboost, shap, scipy, openpyxl

Usage:
    from ml_regression_analysis import MLRegressionAnalysis

    analysis = MLRegressionAnalysis(data_path="data.xlsx", output_dir="output")
    analysis.run_full_analysis(target_column="Extraction rate")

License: MIT
"""

import os
import re
import json
import logging
import argparse
from typing import List, Dict, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from scipy.stats import pearsonr
from scipy.optimize import minimize
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import warnings

# Configure matplotlib for Chinese characters (optional)
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLRegressionAnalysis:
    """
    A comprehensive machine learning regression analysis pipeline.

    This class encapsulates data loading, preprocessing, model training,
    evaluation, visualization, and optimization. It is designed to be
    easily reproducible and suitable for scientific research.

    Attributes:
        data_path (str): Path to the input data file.
        output_dir (str): Root directory for all outputs.
        metrics_dir (str): Directory for metric files.
        figures_dir (str): Directory for figure files.
        models_dir (str): Directory for saved model files.
        data (pd.DataFrame): Raw dataset.
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Target vector.
        target_column (str): Name of the target column.
        X_train, X_test, y_train, y_test: Train/test splits.
        models (Dict[str, Any]): Dictionary of models to be trained.
        predictions (Dict[str, Dict[str, np.ndarray]]): Predictions for each model.
        scores (Dict[str, Dict[str, float]]): Performance metrics for each model.
        models_to_save (Dict[str, Dict[str, Any]]): Model artifacts for saving.
        best_model_name (str): Name of the best model (highest test R²).
        best_model (Dict[str, Any]): Artifacts of the best model.
        feature_ranges (List[Tuple[float, float]]): Min/max ranges for each feature.
    """

    def __init__(self, data_path: str, output_dir: Optional[str] = None,
                 metrics_dir: Optional[str] = None, figures_dir: Optional[str] = None,
                 models_dir: Optional[str] = None):
        """
        Initialize the analysis object.

        Args:
            data_path: Path to the input data file (Excel, CSV, or TXT).
            output_dir: Root output directory (default: same directory as data).
            metrics_dir: Directory for metric files (default: output_dir/metrics).
            figures_dir: Directory for figure files (default: output_dir/figures).
            models_dir: Directory for model files (default: output_dir/models).
        """
        self.data_path = data_path
        self.output_dir = output_dir or os.path.dirname(data_path)
        self._create_directory(self.output_dir)

        self.metrics_dir = metrics_dir or os.path.join(self.output_dir, "metrics")
        self.figures_dir = figures_dir or os.path.join(self.output_dir, "figures")
        self.models_dir = models_dir or os.path.join(self.output_dir, "models")

        for dir_path in [self.metrics_dir, self.figures_dir, self.models_dir]:
            self._create_directory(dir_path)

        # Initialize attributes
        self.data: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[np.ndarray] = None
        self.target_column: Optional[str] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.models: Dict[str, Any] = {}
        self.predictions: Dict[str, Dict[str, np.ndarray]] = {}
        self.scores: Dict[str, Dict[str, float]] = {}
        self.models_to_save: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Dict[str, Any]] = None
        self.feature_ranges: Optional[List[Tuple[float, float]]] = None

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------
    @staticmethod
    def _create_directory(dir_path: str) -> None:
        """Create a directory if it does not exist."""
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            raise

    @staticmethod
    def _normalize_column_name(col: str) -> str:
        """Normalize a column name for matching."""
        return col.strip().lower().replace('（', '(').replace('）', ')').replace('％', '%').replace(' ', '')

    @staticmethod
    def _normalize_feat_name(name: str) -> str:
        """Normalize a feature name by removing non-alphanumeric characters."""
        return re.sub(r'[^\w]', '', name).lower()

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------
    def load_data(self, target_column: Optional[str] = None,
                  feature_columns: Optional[List[str]] = None) -> bool:
        """
        Load data and select target and feature columns.

        Args:
            target_column: Name of target column. If None, attempts auto-detection.
            feature_columns: List of feature column names. If None, all non-target columns are used.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.data = self._read_data_file(self.data_path)
            logger.info(f"Data loaded successfully, shape: {self.data.shape}")
            # Log column names
            logger.info(f"Columns: {list(self.data.columns)}")

            if not self._set_target_column(target_column):
                return False

            self.y = self.data[self.target_column].values

            if not self._set_feature_columns(feature_columns):
                return False

            self.feature_ranges = [(self.X[col].min(), self.X[col].max()) for col in self.X.columns]
            return True
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return False

    def _read_data_file(self, file_path: str) -> pd.DataFrame:
        """Read data file based on extension."""
        file_ext = os.path.splitext(file_path)[1].lower()
        try:
            if file_ext in ['.xlsx', '.xls']:
                try:
                    return pd.read_excel(file_path, engine='openpyxl')
                except:
                    return pd.read_excel(file_path, engine='xlrd')
            elif file_ext == '.csv':
                return pd.read_csv(file_path)
            elif file_ext == '.txt':
                return pd.read_csv(file_path, sep='\t')
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            logger.error(f"Failed to read data file: {e}")
            raise

    def _set_target_column(self, target_column: Optional[str]) -> bool:
        """Set target column, auto-detect if not provided."""
        if target_column:
            if target_column in self.data.columns:
                self.target_column = target_column
                logger.info(f"Target column set to: '{self.target_column}'")
                return True
            else:
                logger.error(f"Specified target column '{target_column}' not found in data.")
                return False

        # Auto-detect common target column names
        possible_target_columns = [
            'Extraction rate', '提取率', 'yield', 'Yield', 'output',
            'Extraction rate(%)', 'Extraction rate(％)', '提取率(%)',
            '提取率(％)', 'Extraction quantity(mg/g)'
        ]
        normalized_columns = {self._normalize_column_name(col): col for col in self.data.columns}
        for col_pattern in possible_target_columns:
            normalized_pattern = self._normalize_column_name(col_pattern)
            if normalized_pattern in normalized_columns:
                self.target_column = normalized_columns[normalized_pattern]
                logger.info(f"Auto-detected target column: '{self.target_column}'")
                return True

        logger.error("No target column detected. Please specify target_column explicitly.")
        return False

    def _set_feature_columns(self, feature_columns: Optional[List[str]]) -> bool:
        """Set feature columns, default to all non-target columns."""
        if feature_columns:
            invalid_features = [f for f in feature_columns if f not in self.data.columns]
            if invalid_features:
                logger.error(f"Specified feature columns not found: {invalid_features}")
                return False
            self.X = self.data[feature_columns]
            logger.info(f"Using specified features: {list(self.X.columns)}")
            return True

        # Default: all columns except target and obvious index columns
        all_columns = set(self.data.columns)
        all_columns.remove(self.target_column)

        # Remove columns that look like indices
        for col in list(all_columns):
            if self._normalize_column_name(col) in ['序号', 'index', 'id', 'no']:
                all_columns.remove(col)
                logger.info(f"Automatically excluded index-like column: '{col}'")

        self.X = self.data[list(all_columns)]
        logger.info(f"Using default features: {list(self.X.columns)}")
        logger.info(f"Feature count: {self.X.shape[1]}, Sample count: {self.X.shape[0]}")
        return True

    # -------------------------------------------------------------------------
    # Data preprocessing
    # -------------------------------------------------------------------------
    def handle_missing_values(self, strategy: str = 'mean') -> bool:
        """
        Handle missing values in the feature matrix.

        Args:
            strategy: One of 'mean', 'median', or 'drop'.

        Returns:
            bool: True if successful.
        """
        if self.X is None:
            logger.error("Data not loaded.")
            return False

        missing_count = self.X.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Detected {missing_count} missing values, applying strategy '{strategy}'.")
            if strategy == 'mean':
                self.X = self.X.fillna(self.X.mean())
            elif strategy == 'median':
                self.X = self.X.fillna(self.X.median())
            elif strategy == 'drop':
                mask = self.X.notnull().all(axis=1)
                self.X = self.X[mask]
                self.y = self.y[mask]
                logger.info(f"Dropped rows with missing values, remaining samples: {len(self.X)}")
            else:
                logger.error(f"Unknown missing value strategy: {strategy}")
                return False
            logger.info("Missing values handled.")
        else:
            logger.info("No missing values detected.")
        return True

    def handle_outliers(self, threshold: float = 3) -> bool:
        """
        Detect and remove outliers using Z-score method.

        Args:
            threshold: Z-score threshold above which a sample is considered an outlier.

        Returns:
            bool: True if successful.
        """
        if self.X is None:
            logger.error("Data not loaded.")
            return False

        z_scores = pd.DataFrame(index=self.X.index)
        for col in self.X.columns:
            std = self.X[col].std()
            if std > 0:
                z_scores[col] = np.abs((self.X[col] - self.X[col].mean()) / std)
            else:
                z_scores[col] = 0  # constant column

        mask = (z_scores < threshold).all(axis=1)
        n_outliers = np.sum(~mask)
        if n_outliers > 0:
            logger.info(f"Removed {n_outliers} outlier samples.")
            self.X = self.X[mask]
            self.y = self.y[mask]
            self.feature_ranges = [(self.X[col].min(), self.X[col].max()) for col in self.X.columns]
        else:
            logger.info("No outliers detected.")
        return True

    def augment_data(self, X: pd.DataFrame, y: np.ndarray,
                     multiplier: int = 6, random_state: int = 48) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Augment dataset by adding small Gaussian noise to original samples.

        Args:
            X: Feature matrix.
            y: Target vector.
            multiplier: Number of total samples per original sample (including original).
            random_state: Random seed.

        Returns:
            Tuple of augmented X and y.
        """
        augmented_X = []
        augmented_y = []
        np.random.seed(random_state)

        feature_stats = {}
        for col in X.columns:
            feature_stats[col] = {
                'min': X[col].min(),
                'max': X[col].max(),
                'std': X[col].std()
            }

        for i in range(len(X)):
            augmented_X.append(X.iloc[i].values)
            augmented_y.append(y[i])

            for _ in range(multiplier - 1):
                new_sample = []
                for col_idx, col in enumerate(X.columns):
                    std = feature_stats[col]['std']
                    noise = np.random.normal(0, std * 0.1) if std > 0 else 0
                    new_value = X.iloc[i, col_idx] + noise
                    new_value = max(feature_stats[col]['min'], new_value)
                    new_value = min(feature_stats[col]['max'], new_value)
                    new_sample.append(new_value)
                augmented_X.append(new_sample)

                noise_y = np.random.normal(0, np.std(y) * 0.05) if np.std(y) > 0 else 0
                new_y = y[i] + noise_y
                augmented_y.append(new_y)

        augmented_X = pd.DataFrame(augmented_X, columns=X.columns)
        augmented_y = np.array(augmented_y)

        # Shuffle
        indices = np.random.permutation(len(augmented_X))
        augmented_X = augmented_X.iloc[indices].reset_index(drop=True)
        augmented_y = augmented_y[indices]

        return augmented_X, augmented_y

    def prepare_data(self, test_size: float = 0.2, random_state: int = 48,
                     use_augmentation: bool = True, handle_outliers: bool = True,
                     outlier_threshold: float = 3) -> bool:
        """
        Prepare train/test data with preprocessing steps.

        Args:
            test_size: Proportion of test set.
            random_state: Random seed for reproducibility.
            use_augmentation: Whether to augment data if sample size < 100.
            handle_outliers: Whether to remove outliers.
            outlier_threshold: Outlier Z-score threshold.

        Returns:
            bool: True if successful.
        """
        if self.X is None or self.y is None:
            logger.error("Data not loaded.")
            return False

        if not self.handle_missing_values():
            return False

        if handle_outliers and not self.handle_outliers(threshold=outlier_threshold):
            return False

        if use_augmentation and len(self.X) < 100:
            logger.info("Sample size < 100, applying data augmentation.")
            X_aug, y_aug = self.augment_data(self.X, self.y, random_state=random_state)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_aug, y_aug, test_size=test_size, random_state=random_state
            )
            logger.info(f"Augmented data split: train={len(self.X_train)}, test={len(self.X_test)}")
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state
            )
            logger.info(f"Data split: train={len(self.X_train)}, test={len(self.X_test)}")

        return True

    # -------------------------------------------------------------------------
    # Model definition and training
    # -------------------------------------------------------------------------
    def define_models(self, custom_models: Optional[Dict[str, Any]] = None,
                      include_all: bool = True) -> bool:
        """
        Define models to be trained.

        Args:
            custom_models: Dictionary of custom models (name -> estimator).
            include_all: Whether to include default models.

        Returns:
            bool: True if successful.
        """
        base_models = {}
        if include_all:
            base_models = {
                "Linear": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "DT": DecisionTreeRegressor(max_depth=3, random_state=42),
                "RF": RandomForestRegressor(random_state=42),
                "AdaBoost": AdaBoostRegressor(random_state=42),
                "BP": MLPRegressor(max_iter=5000, random_state=42),
                "Xgboost": XGBRegressor(
                    n_estimators=280, max_depth=6, learning_rate=0.16,
                    gamma=0.93, reg_alpha=0.86, reg_lambda=0.13,
                    min_child_weight=7, subsample=0.9, colsample_bytree=0.8,
                    random_state=681
                ),
                "SVM": SVR(kernel='rbf', C=100, epsilon=0.1),
                "GBDT": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=421),
            }

        if custom_models:
            self.models = {**base_models, **custom_models}
        else:
            self.models = base_models

        logger.info(f"Defined {len(self.models)} models.")
        return True

    def train_models(self, use_cv: bool = False, cv_folds: int = 5) -> bool:
        """
        Train all defined models.

        Args:
            use_cv: Whether to perform cross-validation.
            cv_folds: Number of CV folds.

        Returns:
            bool: True if successful.
        """
        if self.X_train is None or self.y_train is None:
            logger.error("Data not prepared.")
            return False

        scaled_models = ["BP", "SVM"]  # models requiring feature scaling

        for name, model in self.models.items():
            logger.info(f"Training model: {name}")
            try:
                if name in scaled_models:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(self.X_train)
                    X_test_scaled = scaler.transform(self.X_test)
                    model.fit(X_train_scaled, self.y_train)
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)

                    if use_cv:
                        cv_scores = cross_val_score(model, X_train_scaled, self.y_train, cv=cv_folds, scoring='r2')
                        logger.info(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

                    self.models_to_save[name] = {
                        "model": model,
                        "scaler": scaler,
                        "X_train": X_train_scaled,
                        "X_test": X_test_scaled
                    }
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred_train = model.predict(self.X_train)
                    y_pred_test = model.predict(self.X_test)

                    if use_cv:
                        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv_folds, scoring='r2')
                        logger.info(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

                    self.models_to_save[name] = {
                        "model": model,
                        "scaler": None,
                        "X_train": self.X_train,
                        "X_test": self.X_test
                    }

                self.predictions[name] = {
                    "train": y_pred_train,
                    "test": y_pred_test
                }
            except Exception as e:
                logger.error(f"  Model {name} failed to train: {e}")

        logger.info("Model training completed.")
        return True

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    def evaluate_models(self) -> bool:
        """
        Evaluate model performance and select best model.

        Returns:
            bool: True if successful.
        """
        if not self.predictions:
            logger.error("No predictions available.")
            return False

        for name, preds in self.predictions.items():
            self.scores[name] = {
                "R2_train": r2_score(self.y_train, preds["train"]),
                "RMSE_train": np.sqrt(mean_squared_error(self.y_train, preds["train"])),
                "MAE_train": mean_absolute_error(self.y_train, preds["train"]),
                "R2_test": r2_score(self.y_test, preds["test"]),
                "RMSE_test": np.sqrt(mean_squared_error(self.y_test, preds["test"])),
                "MAE_test": mean_absolute_error(self.y_test, preds["test"])
            }

            logger.info(f"{name}: train R²={self.scores[name]['R2_train']:.4f}, "
                        f"test R²={self.scores[name]['R2_test']:.4f}")

        self.best_model_name = max(self.scores.items(), key=lambda x: x[1]['R2_test'])[0]
        logger.info(f"Best model: {self.best_model_name} (test R²={self.scores[self.best_model_name]['R2_test']:.4f})")
        self.best_model = self.models_to_save[self.best_model_name]

        # Save evaluation metrics to Excel
        results_df = pd.DataFrame.from_dict(self.scores, orient='index')
        results_path = os.path.join(self.metrics_dir, "model_comparison.xlsx")
        results_df.to_excel(results_path, index=True)
        logger.info(f"Model evaluation results saved to: {results_path}")

        return True

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    def _create_subplots(self, n_items: int, base_figsize: Tuple[int, int] = (5, 4),
                         max_cols: int = 4) -> Tuple[plt.Figure, int, int]:
        """Create a subplot grid."""
        cols = min(max_cols, n_items)
        rows = (n_items + cols - 1) // cols
        figsize = (base_figsize[0] * cols, base_figsize[1] * rows)
        fig = plt.figure(figsize=figsize)
        return fig, rows, cols

    def plot_fit_curves(self) -> bool:
        """Plot true vs predicted values for all models (train and test)."""
        if not self.predictions:
            logger.error("No predictions available.")
            return False

        # Training set
        fig, rows, cols = self._create_subplots(len(self.predictions))
        for i, (name, preds) in enumerate(self.predictions.items(), 1):
            plt.subplot(rows, cols, i)
            plt.plot(self.y_train, label='True', marker='o', markersize=3)
            plt.plot(preds["train"], label='Predicted', marker='x', markersize=3)
            plt.xlabel('Sample')
            plt.ylabel('Value')
            plt.title(f'{name} - Train')
            plt.legend()
        plt.tight_layout()
        train_path = os.path.join(self.figures_dir, "fit_curves_train.png")
        plt.savefig(train_path, dpi=300)
        plt.close()
        logger.info(f"Train fit curves saved to: {train_path}")

        # Test set
        fig, rows, cols = self._create_subplots(len(self.predictions))
        for i, (name, preds) in enumerate(self.predictions.items(), 1):
            plt.subplot(rows, cols, i)
            plt.plot(self.y_test, label='True', marker='o', markersize=3)
            plt.plot(preds["test"], label='Predicted', marker='x', markersize=3)
            plt.xlabel('Sample')
            plt.ylabel('Value')
            plt.title(f'{name} - Test')
            plt.legend()
        plt.tight_layout()
        test_path = os.path.join(self.figures_dir, "fit_curves_test.png")
        plt.savefig(test_path, dpi=300)
        plt.close()
        logger.info(f"Test fit curves saved to: {test_path}")

        return True

    def plot_r2_comparison(self) -> bool:
        """Plot R² comparison bar chart."""
        if not self.scores:
            logger.error("No scores available.")
            return False

        model_names = list(self.scores.keys())
        r2_train = [self.scores[n]["R2_train"] for n in model_names]
        r2_test = [self.scores[n]["R2_test"] for n in model_names]

        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        index = np.arange(len(model_names))

        plt.bar(index, r2_train, bar_width, color='skyblue', label='Train R²')
        plt.bar(index + bar_width, r2_test, bar_width, color='orange', label='Test R²')

        for i in range(len(model_names)):
            plt.text(index[i] - 0.02, r2_train[i] + 0.01, f'{r2_train[i]:.2f}', fontsize=9)
            plt.text(index[i] + bar_width - 0.02, r2_test[i] + 0.01, f'{r2_test[i]:.2f}', fontsize=9)

        plt.xlabel('Model')
        plt.ylabel('R²')
        plt.title('R² Comparison')
        plt.xticks(index + bar_width / 2, model_names, rotation=45, ha='right')
        plt.legend()

        path = os.path.join(self.figures_dir, "r2_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info(f"R² comparison saved to: {path}")

        return True

    def plot_scatter_plots(self) -> bool:
        """Plot scatter plots of true vs predicted values."""
        if not self.predictions:
            logger.error("No predictions available.")
            return False

        # Training set
        fig, rows, cols = self._create_subplots(len(self.predictions))
        for i, (name, preds) in enumerate(self.predictions.items(), 1):
            plt.subplot(rows, cols, i)
            plt.scatter(self.y_train, preds["train"], alpha=0.6)
            min_val = min(self.y_train.min(), preds["train"].min())
            max_val = max(self.y_train.max(), preds["train"].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            plt.xlabel('True (train)')
            plt.ylabel('Predicted (train)')
            r_val, _ = pearsonr(self.y_train, preds["train"])
            plt.title(f'{name} - R = {r_val:.2f}')
        plt.tight_layout()
        path = os.path.join(self.figures_dir, "scatter_train.png")
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info(f"Train scatter plots saved to: {path}")

        # Test set
        fig, rows, cols = self._create_subplots(len(self.predictions))
        for i, (name, preds) in enumerate(self.predictions.items(), 1):
            plt.subplot(rows, cols, i)
            plt.scatter(self.y_test, preds["test"], alpha=0.6)
            min_val = min(self.y_test.min(), preds["test"].min())
            max_val = max(self.y_test.max(), preds["test"].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            plt.xlabel('True (test)')
            plt.ylabel('Predicted (test)')
            r_val, _ = pearsonr(self.y_test, preds["test"])
            plt.title(f'{name} - R = {r_val:.2f}')
        plt.tight_layout()
        path = os.path.join(self.figures_dir, "scatter_test.png")
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info(f"Test scatter plots saved to: {path}")

        return True

    def plot_residuals(self) -> bool:
        """Plot residual plots for test set."""
        if not self.predictions:
            logger.error("No predictions available.")
            return False

        fig, rows, cols = self._create_subplots(len(self.predictions))
        for i, (name, preds) in enumerate(self.predictions.items(), 1):
            plt.subplot(rows, cols, i)
            residuals = self.y_test - preds["test"]
            plt.scatter(preds["test"], residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted')
            plt.ylabel('Residual')
            plt.title(f'{name} - Residuals')
        plt.tight_layout()
        path = os.path.join(self.figures_dir, "residuals.png")
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info(f"Residual plots saved to: {path}")

        return True

    def plot_correlation_heatmap(self) -> bool:
        """Plot feature correlation heatmap."""
        if self.X is None:
            logger.error("Data not loaded.")
            return False

        plt.figure(figsize=(10, 8))
        corr = self.X.corr()
        plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Correlation coefficient')
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
        plt.yticks(range(len(corr.columns)), corr.columns)

        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                plt.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                         color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')

        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        path = os.path.join(self.figures_dir, "correlation_heatmap.png")
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info(f"Correlation heatmap saved to: {path}")

        return True

    # -------------------------------------------------------------------------
    # Results saving and interpretation
    # -------------------------------------------------------------------------
    def save_test_predictions(self) -> bool:
        """Save test set predictions to Excel."""
        if not self.predictions:
            logger.error("No predictions available.")
            return False

        pred_df = pd.DataFrame({'True': self.y_test})
        for name, preds in self.predictions.items():
            pred_df[f'{name}_pred'] = preds["test"]

        path = os.path.join(self.metrics_dir, "test_predictions.xlsx")
        pred_df.to_excel(path, index=False)
        logger.info(f"Test predictions saved to: {path}")
        return True

    def save_best_model(self) -> bool:
        """Save the best model and perform SHAP/PDP analyses."""
        if not self.best_model:
            logger.error("No best model selected.")
            return False

        try:
            model = self.best_model["model"]
            scaler = self.best_model["scaler"]
            X_test = self.best_model["X_test"]
            model_name = self.best_model_name
            feature_names = list(self.X.columns)

            model_path = os.path.join(self.models_dir, f"{model_name}_model.pkl")
            scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.pkl")

            joblib.dump(model, model_path)
            logger.info(f"Best model saved to: {model_path}")

            if scaler:
                joblib.dump(scaler, scaler_path)
                logger.info(f"Scaler saved to: {scaler_path}")

            self._perform_shap_analysis(model, X_test, model_name, feature_names)
            self._perform_pdp_analysis(model, X_test, model_name, feature_names)

            return True
        except Exception as e:
            logger.error(f"Failed to save model or generate analysis: {e}")
            return False

    def _perform_shap_analysis(self, model: Any, X_test: Union[pd.DataFrame, np.ndarray],
                               model_name: str, feature_names: List[str]) -> None:
        """Perform SHAP analysis and save plots."""
        try:
            logger.info(f"Performing SHAP analysis for {model_name}...")
            explainer = shap.Explainer(model.predict, X_test)
            shap_values = explainer(X_test)

            # Summary bar plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
            plt.title(f'{model_name} SHAP Feature Importance')
            plt.tight_layout()
            path = os.path.join(self.figures_dir, f"{model_name}_shap_summary.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP summary saved to: {path}")

            # Bee swarm plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.title(f'{model_name} SHAP Analysis')
            plt.tight_layout()
            path = os.path.join(self.figures_dir, f"{model_name}_shap_bee.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP bee plot saved to: {path}")

            self._save_feature_importance(shap_values, feature_names, model_name)
        except Exception as e:
            logger.error(f"SHAP analysis failed for {model_name}: {e}")

    def _save_feature_importance(self, shap_values: Any, feature_names: List[str], model_name: str) -> None:
        """Save feature importance table based on SHAP variance."""
        shap_variances = np.var(shap_values.values, axis=0)
        total_variance = np.sum(shap_variances)

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP variance': shap_variances,
            'Importance (%)': (shap_variances / total_variance * 100).round(2)
        }).sort_values('SHAP variance', ascending=False)

        path = os.path.join(self.metrics_dir, f"{model_name}_feature_importance.xlsx")
        importance_df.to_excel(path, index=False)
        logger.info(f"Feature importance saved to: {path}")

    def _perform_pdp_analysis(self, model: Any, X_test: Union[pd.DataFrame, np.ndarray],
                              model_name: str, feature_names: List[str]) -> None:
        """Perform Partial Dependence Plot (PDP) analysis."""
        try:
            logger.info(f"Performing PDP analysis for {model_name}...")
            n_features = min(5, len(feature_names))

            fig, ax = plt.subplots(figsize=(10, 6))
            PartialDependenceDisplay.from_estimator(
                estimator=model,
                X=X_test,
                features=list(range(n_features)),
                feature_names=feature_names,
                ax=ax,
                grid_resolution=50
            )
            plt.title(f'{model_name} Partial Dependence Plots')
            plt.tight_layout()
            path = os.path.join(self.figures_dir, f"{model_name}_pdp.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"PDP saved to: {path}")
        except Exception as e:
            logger.error(f"PDP analysis failed for {model_name}: {e}")

    # -------------------------------------------------------------------------
    # Feature interaction analysis
    # -------------------------------------------------------------------------
    def plot_feature_interactions(self) -> bool:
        """Plot 3D surface plots for pairwise interactions of top-3 features."""
        if not self.best_model:
            logger.error("No best model selected.")
            return False

        try:
            model = self.best_model["model"]
            model_name = self.best_model_name
            feature_names = [col.strip() for col in self.X.columns]
            X_test = self.best_model["X_test"]

            top_features = self._get_top_features(model, model_name, feature_names)
            if len(top_features) < 3:
                logger.error("Insufficient features for interaction analysis (need >= 3).")
                return False

            matched_features = self._match_features(top_features, feature_names)
            if len(matched_features) < 3:
                logger.error("Could not match enough features for interaction analysis.")
                return False

            logger.info(f"Matched features for interaction: {matched_features}")
            feature_indices = [feature_names.index(f) for f in matched_features]
            interaction_pairs = [(feature_indices[0], feature_indices[1]),
                                 (feature_indices[0], feature_indices[2]),
                                 (feature_indices[1], feature_indices[2])]

            self._plot_interaction_surfaces(model, X_test, interaction_pairs, feature_names, model_name)
            return True
        except Exception as e:
            logger.error(f"Feature interaction analysis failed: {e}")
            return False

    def _get_top_features(self, model: Any, model_name: str, feature_names: List[str]) -> List[str]:
        """Get top-3 important features (prefer SHAP, fallback to model built-in importance)."""
        top_features = []
        importance_path = os.path.join(self.metrics_dir, f"{model_name}_feature_importance.xlsx")

        if os.path.exists(importance_path):
            try:
                importance_df = pd.read_excel(importance_path)
                top_features = importance_df["Feature"].head(3).tolist()
                logger.info(f"Top features from SHAP: {top_features}")
                return top_features
            except Exception as e:
                logger.warning(f"Failed to read SHAP importance: {e}")

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_indices = importances.argsort()[-3:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            logger.info(f"Top features from built-in importance: {top_features}")
            return top_features

        top_features = feature_names[:3]
        logger.warning(f"Using first three features as fallback: {top_features}")
        return top_features

    def _match_features(self, top_features: List[str], all_features: List[str]) -> List[str]:
        """Match feature names with normalization."""
        norm_all = {self._normalize_feat_name(f): f for f in all_features}
        norm_top = [self._normalize_feat_name(f) for f in top_features]

        matched = []
        for nt in norm_top:
            if nt in norm_all:
                matched.append(norm_all[nt])
            else:
                # fuzzy match
                for na, orig in norm_all.items():
                    if nt in na or na in nt:
                        matched.append(orig)
                        break
        return matched

    def _plot_interaction_surfaces(self, model: Any, X_test: Union[pd.DataFrame, np.ndarray],
                                   interaction_pairs: List[Tuple[int, int]], feature_names: List[str],
                                   model_name: str) -> None:
        """Plot 3D interaction surfaces."""
        fig = plt.figure(figsize=(18, 6))
        for i, (f1_idx, f2_idx) in enumerate(interaction_pairs):
            try:
                f1_name = feature_names[f1_idx]
                f2_name = feature_names[f2_idx]

                results = partial_dependence(
                    estimator=model, X=X_test, features=[f1_idx, f2_idx], grid_resolution=20
                )

                grids = results["grid_values"]
                predictions = results["average"]

                xx, yy = np.meshgrid(grids[0], grids[1])
                z = predictions.reshape(xx.shape)
                z = np.nan_to_num(z)

                ax = fig.add_subplot(1, 3, i + 1, projection='3d')
                surf = ax.plot_surface(xx, yy, z, cmap='viridis', alpha=0.8, edgecolor='none')
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=self.target_column)
                ax.set_xlabel(f1_name)
                ax.set_ylabel(f2_name)
                ax.set_zlabel(self.target_column)
                ax.set_title(f'{f1_name} vs {f2_name}')
            except Exception as e:
                logger.error(f"Failed to plot interaction: {e}")

        path = os.path.join(self.figures_dir, f"{model_name}_interaction_surfaces.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Interaction surfaces saved to: {path}")

    # -------------------------------------------------------------------------
    # Parameter optimization
    # -------------------------------------------------------------------------
    def find_optimal_parameters(self) -> bool:
        """Find optimal feature values that maximize the predicted target."""
        if not self.best_model or self.feature_ranges is None:
            logger.error("No best model or feature ranges available.")
            return False

        try:
            model = self.best_model["model"]
            scaler = self.best_model["scaler"]
            model_name = self.best_model_name
            feature_names = list(self.X.columns)

            logger.info(f"Optimizing parameters using {model_name} model...")

            # Adjust bounds to avoid degenerate intervals
            bounds = []
            for lower, upper in self.feature_ranges:
                if lower == upper:
                    eps = 1e-6
                    bounds.append((lower - eps, upper + eps))
                else:
                    bounds.append((lower, upper))

            result = minimize(
                lambda params: self._objective_function(params, model, scaler),
                [self.X[col].mean() for col in self.X.columns],
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )

            if not result.success:
                logger.error(f"Optimization failed: {result.message}")
                return False

            self._process_optimization_results(result, feature_names, model_name)
            return True
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return False

    def _objective_function(self, params: np.ndarray, model: Any, scaler: Optional[StandardScaler]) -> float:
        """Negative prediction (for minimization)."""
        params_array = np.array(params).reshape(1, -1)
        if scaler:
            params_scaled = scaler.transform(params_array)
            prediction = model.predict(params_scaled)
        else:
            prediction = model.predict(params_array)
        return -prediction[0]

    def _process_optimization_results(self, result: Any, feature_names: List[str], model_name: str) -> None:
        """Process and save optimization results."""
        optimal_params = result.x
        max_predicted = -result.fun

        opt_df = pd.DataFrame({
            'Feature': feature_names,
            'Optimal value': optimal_params,
            'Min': [b[0] for b in self.feature_ranges],
            'Max': [b[1] for b in self.feature_ranges]
        })

        path = os.path.join(self.metrics_dir, f"{model_name}_optimal_parameters.xlsx")
        opt_df.to_excel(path, index=False)
        logger.info(f"Optimal parameters saved to: {path}")

        # Print top-3 features' optimal values
        self._print_top3_optimal_features(opt_df, model_name, feature_names)
        logger.info(f"Maximum predicted {self.target_column}: {max_predicted:.4f}")

    def _print_top3_optimal_features(self, opt_df: pd.DataFrame, model_name: str, feature_names: List[str]) -> None:
        """Print optimal values for the three most important features."""
        try:
            importance_path = os.path.join(self.metrics_dir, f"{model_name}_feature_importance.xlsx")
            if os.path.exists(importance_path):
                importance_df = pd.read_excel(importance_path)
                merged = pd.merge(
                    importance_df[['Feature', 'Importance (%)']],
                    opt_df[['Feature', 'Optimal value']],
                    on='Feature'
                ).sort_values('Importance (%)', ascending=False)

                logger.info("\nTop-3 important features at optimum:")
                top3 = merged.head(3)
                for _, row in top3.iterrows():
                    logger.info(f"  {row['Feature']}: {row['Optimal value']:.4f} (importance {row['Importance (%)']}%)")

                # Save top3
                path = os.path.join(self.metrics_dir, f"{model_name}_top3_optimal.xlsx")
                top3.to_excel(path, index=False)
                logger.info(f"Top-3 optimal values saved to: {path}")
                return
        except Exception as e:
            logger.warning(f"Failed to get importance for top-3: {e}")

        # Fallback: first three features
        logger.info("\nTop-3 features (by column order) at optimum:")
        for i in range(min(3, len(feature_names))):
            logger.info(f"  {feature_names[i]}: {opt_df.iloc[i]['Optimal value']:.4f}")

    # -------------------------------------------------------------------------
    # Utility: save configuration
    # -------------------------------------------------------------------------
    def save_config(self, **kwargs) -> None:
        """Save configuration parameters to a JSON file for reproducibility."""
        config = {
            'data_path': self.data_path,
            'target_column': self.target_column,
            'feature_columns': list(self.X.columns) if self.X is not None else None,
            **kwargs
        }
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info(f"Configuration saved to: {config_path}")

    # -------------------------------------------------------------------------
    # Main pipeline
    # -------------------------------------------------------------------------
    def run_full_analysis(self, target_column: Optional[str] = None,
                          feature_columns: Optional[List[str]] = None,
                          use_cv: bool = False,
                          use_augmentation: bool = True,
                          handle_outliers: bool = True,
                          outlier_threshold: float = 3,
                          random_state: int = 48,
                          test_size: float = 0.2) -> bool:
        """
        Run the complete analysis pipeline.

        Args:
            target_column: Target column name.
            feature_columns: Feature column names.
            use_cv: Whether to perform cross-validation.
            use_augmentation: Whether to apply data augmentation.
            handle_outliers: Whether to remove outliers.
            outlier_threshold: Outlier Z-score threshold.
            random_state: Random seed.
            test_size: Test set ratio.

        Returns:
            bool: True if successful.
        """
        logger.info("===== Starting machine learning regression analysis =====")

        # Save configuration before running (as much as known)
        self.save_config(
            use_cv=use_cv,
            use_augmentation=use_augmentation,
            handle_outliers=handle_outliers,
            outlier_threshold=outlier_threshold,
            random_state=random_state,
            test_size=test_size
        )

        success = (
            self.load_data(target_column, feature_columns) and
            self.prepare_data(test_size=test_size, random_state=random_state,
                              use_augmentation=use_augmentation,
                              handle_outliers=handle_outliers,
                              outlier_threshold=outlier_threshold) and
            self.define_models() and
            self.train_models(use_cv=use_cv) and
            self.evaluate_models() and
            self.plot_fit_curves() and
            self.plot_r2_comparison() and
            self.plot_scatter_plots() and
            self.plot_residuals() and
            self.plot_correlation_heatmap() and
            self.save_test_predictions() and
            self.save_best_model() and
            self.plot_feature_interactions() and
            self.find_optimal_parameters()
        )

        if success:
            logger.info("===== Analysis completed successfully =====")
        else:
            logger.error("===== Analysis terminated due to errors =====")
        return success


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Machine Learning Regression Analysis')
    parser.add_argument('--data_path', required=True, help='Path to input data file')
    parser.add_argument('--output_dir', help='Output directory (default: same as data)')
    parser.add_argument('--target_column', help='Target column name')
    parser.add_argument('--feature_columns', nargs='+', help='Feature column names')
    parser.add_argument('--use_cv', action='store_true', help='Use cross-validation')
    parser.add_argument('--no_augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--no_outlier_removal', action='store_true', help='Disable outlier removal')
    parser.add_argument('--outlier_threshold', type=float, default=3.0, help='Outlier Z-score threshold')
    parser.add_argument('--random_state', type=int, default=48, help='Random seed')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set ratio')
    parser.add_argument('--metrics_dir', help='Metrics directory')
    parser.add_argument('--figures_dir', help='Figures directory')
    parser.add_argument('--models_dir', help='Models directory')
    return parser.parse_args()


def main():
    """Command-line entry point."""
    args = parse_arguments()
    output_dir = args.output_dir or os.path.dirname(args.data_path)

    analysis = MLRegressionAnalysis(
        data_path=args.data_path,
        output_dir=output_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        models_dir=args.models_dir
    )

    analysis.run_full_analysis(
        target_column=args.target_column,
        feature_columns=args.feature_columns,
        use_cv=args.use_cv,
        use_augmentation=not args.no_augmentation,
        handle_outliers=not args.no_outlier_removal,
        outlier_threshold=args.outlier_threshold,
        random_state=args.random_state,
        test_size=args.test_size
    )

    print(f"\nAll results saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()