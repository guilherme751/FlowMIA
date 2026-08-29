"""
Utility evaluation for synthetic tabular data.

Implements two standard protocols used to benchmark how well a synthetic
dataset preserves the predictive signal of the original:

- **RTR** (Real-Train / Real-Test): classifier trained on real data.
- **TSTR** (Synthetic-Train / Real-Test): classifier trained on synthetic data.

Both are evaluated on the same held-out test set, so scores are directly
comparable. A TSTR performance close to RTR indicates high utility.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class Utility:
    """
    RTR / TSTR utility evaluator for synthetic tabular data.

    Preprocessing (RobustScaler + OneHotEncoder) is fitted independently
    inside each protocol pipeline, which mirrors real-world usage where
    the synthetic consumer has no access to the original scaler.

    IP address columns are treated as categorical features (passed through
    OneHotEncoder) since they are not ordinal in the utility context.

    Args:
        classifiers: List of scikit-learn compatible classifier instances.
        train: Real training (member) dataframe.
        test: Held-out test dataframe, shared by both protocols.
        synth: Synthetic training dataframe.
        categorical_cols: Names of categorical feature columns.
        numerical_cols: Names of numerical feature columns.
        ip_cols: Names of IP address columns (treated as categorical).
        label_col: Name of the target column.
    """

    def __init__(
        self,
        classifiers: list,
        train,
        test,
        synth,
        categorical_cols: list,
        numerical_cols: list,
        ip_cols: list,
        label_col: str,
    ):
        self.classifiers = classifiers
        self.label_col = label_col
        self.num_cols = numerical_cols
        # IP columns have no natural ordering, so they go through OHE
        self.cat_cols = categorical_cols + ip_cols

        feature_cols = self.num_cols + self.cat_cols
        self.X_train = train[feature_cols]
        self.y_train = train[label_col]
        self.X_synth = synth[feature_cols]
        self.y_synth = synth[label_col]
        self.X_test = test[feature_cols]
        self.y_test = test[label_col]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self, clf) -> Pipeline:
        """
        Build a fresh sklearn Pipeline with preprocessing and a classifier.

        A new preprocessor is created each call so that RTR and TSTR pipelines
        fit their own scalers independently.

        Args:
            clf: An unfitted scikit-learn classifier.

        Returns:
            Pipeline ready to be fitted.
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", RobustScaler(), self.num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols),
            ]
        )
        return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])

    def _compute_metrics(self, y_true, y_pred) -> dict:
        """
        Compute classification metrics for a single prediction set.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary with Accuracy, Precision, Recall, and F1-Score.
        """
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    # ------------------------------------------------------------------
    # Protocols
    # ------------------------------------------------------------------

    def rtr(self, clf) -> dict:
        """
        Train on real data, test on real data (RTR).

        Provides the upper-bound reference performance for a given classifier.

        Args:
            clf: An unfitted scikit-learn classifier.

        Returns:
            Metrics dictionary (Accuracy, Precision, Recall, F1-Score).
        """
        pipeline = self._build_pipeline(clf)
        pipeline.fit(self.X_train, self.y_train)
        return self._compute_metrics(self.y_test, pipeline.predict(self.X_test))

    def tstr(self, clf) -> dict:
        """
        Train on synthetic data, test on real data (TSTR).

        The closer these scores are to RTR, the higher the utility of the
        synthetic dataset for downstream classification tasks.

        Args:
            clf: An unfitted scikit-learn classifier.

        Returns:
            Metrics dictionary (Accuracy, Precision, Recall, F1-Score).
        """
        pipeline = self._build_pipeline(clf)
        pipeline.fit(self.X_synth, self.y_synth)
        return self._compute_metrics(self.y_test, pipeline.predict(self.X_test))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """
        Run RTR and TSTR for every classifier and collect results.

        Returns:
            Nested dictionary of the form
            ``{classifier_name: {"RTR": metrics, "TSTR": metrics}}``.
        """
        results = {}
        for clf in self.classifiers:
            name = clf.__class__.__name__
            print(f"  [{name}] running RTR...")
            rtr = self.rtr(clf)
            print(f"  [{name}] running TSTR...")
            tstr = self.tstr(clf)
            results[name] = {"RTR": rtr, "TSTR": tstr}
        return results

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot_utility(self, utility_dict: dict, save_path: str) -> None:
        """
        Save a 2×2 grid comparing RTR and TSTR scores across classifiers.

        Each subplot corresponds to one metric. RTR scores are shown as
        scatter points overlaid on TSTR bars, making deviations easy to spot.

        Args:
            utility_dict: Output of :meth:`evaluate`.
            save_path: Root directory; the plot is written to
                ``save_path/plots/utility.pdf``.
        """
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        classifiers = list(utility_dict.keys())
        x = np.arange(len(classifiers))
        bar_width = 0.6

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            tstr_values = [utility_dict[clf]["TSTR"][metric] for clf in classifiers]
            rtr_values = [utility_dict[clf]["RTR"][metric] for clf in classifiers]

            ax = axes[i]
            ax.bar(x, tstr_values, width=bar_width, alpha=0.7, label="TSTR")
            ax.scatter(x, rtr_values, zorder=3, label="RTR")
            ax.set_title(metric)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            if i >= 2:
                ax.set_xticks(x)
                ax.set_xticklabels(classifiers, rotation=30, ha="right")
            if i == 0:
                ax.legend()

        plt.tight_layout()

        plot_dir = os.path.join(save_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        out_path = os.path.join(plot_dir, "utility.pdf")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")