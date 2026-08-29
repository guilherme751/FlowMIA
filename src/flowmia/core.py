"""
FlowMIA: A framework for privacy evaluation of synthetic tabular data.

Supports Membership Inference Attacks (MIA) via:
    - FlowMIA-GAN: discriminator-based MIA using a trained GAN
    - DOMIAS: density-ratio MIA using normalizing flows (BNAF)
    - DCR: distance-to-closest-record MIA using nearest neighbors
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from flowmia.attacks import get_attack
from flowmia.config import load_config

from flowmia.utility import Utility
from flowmia.fidelity import fidelity_compute, plotFidelity


class FlowMIA:
    """
    Orchestrates privacy evaluation of a synthetic dataset against real training data.

    Preprocessing is handled once here and shared across all attack methods.
    Numerical columns are scaled with RobustScaler, categorical columns are
    one-hot encoded, and IP address columns are normalized to [0, 1] by
    dividing by 2**32 - 1.

    Args:
        config (dict): Configuration dictionary with the following keys:

            - member_path (str): Path to the training (member) CSV.
            - non_member_path (str): Path to the held-out (non-member) CSV.
            - synth_path (str): Path to the synthetic data CSV.
            - test_path (str): Path to the utility test CSV.
            - save_path (str): Directory where outputs and checkpoints are saved.
            - categorical_cols (list[str]): Names of categorical feature columns.
            - numerical_cols (list[str]): Names of numerical feature columns.
            - ip_cols (list[str]): Names of IP address columns.
            - label_col (str): Name of the label/target column.
            - use_wgan (bool): Whether to use WGAN-GP instead of standard GAN.
            - batch_size (int): Batch size for GAN training.
            - num_epochs (int): Number of GAN training epochs.
            - fcheckpoint (int): Checkpoint save frequency (in epochs).
    """

    def __init__(self, config: dict):
        self.config = config


        os.makedirs(config["save_path"], exist_ok=True)
        self.save_path = config["save_path"]

        self.categorical_cols = config["categorical_cols"]
        self.numerical_cols = config["numerical_cols"]
        self.ip_cols = config["ip_cols"]
        self.label_col = config["label_col"]
        self.use_wgan = config["use_wgan"]

        self.member = pd.read_csv(config["member_path"])[self.ip_cols + self.numerical_cols + self.categorical_cols + [self.label_col]]
        self.non_member = pd.read_csv(config["non_member_path"])[self.ip_cols + self.numerical_cols + self.categorical_cols + [self.label_col]]
        self.synth = pd.read_csv(config["synth_path"])[self.ip_cols + self.numerical_cols + self.categorical_cols + [self.label_col]]
        self.util_test = pd.read_csv(config["test_path"])[self.ip_cols + self.numerical_cols + self.categorical_cols + [self.label_col]]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # --------------- preprocessor ---------------
        # Fits on synthetic data so the attack sees the same feature space
        # as the generator. IP columns are handled separately (see _transform).
        all_cat = self.categorical_cols 
        all_categories = []

        for col in all_cat:
            # pega categorias do REAL (não do synth)
            cats = pd.concat([self.member[col], self.non_member[col]]).unique()
            all_categories.append(sorted(cats))

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numerical_cols),
                ("cat", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    categories=all_categories
                ), all_cat),
            ]
        )
        fit_data = pd.concat([self.synth, self.member], axis=0)
        self.preprocessor.fit(fit_data)

        # Pre-transform every split once so downstream methods receive arrays
        self.X_member = self._transform(self.member)
        self.X_non_member = self._transform(self.non_member)
        self.X_synth = self._transform(self.synth)
        print(self.X_member.shape, self.X_non_member.shape, self.X_synth.shape)

    @classmethod
    def from_yaml(cls, path) -> "FlowMIA":
        """
        Construct FlowMIA from a YAML config file.

        Args:
            path: Path to a YAML file with the same keys as the ``config``
                dict accepted by the regular constructor. See
                :class:`flowmia.config.FlowMIAConfig`.

        Returns:
            A constructed FlowMIA instance.
        """
        config = load_config(path)
        return cls(config=config.to_dict())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted preprocessor and append normalized IP columns.

        Args:
            df: Raw input dataframe.

        Returns:
            2-D float32 array ready to feed into attack models.
        """
        X = self.preprocessor.transform(df).astype(np.float32)
        if self.ip_cols:
            ips = (df[self.ip_cols].values / (2**32 - 1)).astype(np.float32)
            X = np.hstack([X, ips])
        return X

    # ------------------------------------------------------------------
    # Attack methods
    # ------------------------------------------------------------------

    def _run_attack(self, name: str, fit_kwargs: dict = None, attack_kwargs: dict = None):
        """
        Instantiate a registered attack by name and run fit() + attack() on it.

        Args:
            name: Name the attack was registered under (see
                :func:`flowmia.attacks.list_attacks`).
            fit_kwargs: Keyword arguments forwarded to the attack's ``fit()``.
            attack_kwargs: Keyword arguments forwarded to the attack's ``attack()``.

        Returns:
            Tuple of (attack instance, :class:`~flowmia.attacks.AttackResult`).
        """
        attack_cls = get_attack(name)
        attack = attack_cls(device=self.device)
        attack.fit(self.X_member, self.X_non_member, self.X_synth, **(fit_kwargs or {}))
        result = attack.attack(self.X_member, self.X_non_member, **(attack_kwargs or {}))
        return attack, result

    def flowmiagan(self, pre_trained_model: str = None, plot: bool = False, test_size=1000) -> dict:
        """
        Run the FlowMIA-GAN membership inference attack.

        Trains (or loads) a GAN on the synthetic data, then uses the
        discriminator's confidence scores to separate members from non-members.

        Args:
            pre_trained_model: Path to a saved checkpoint. When provided,
                training is skipped and the checkpoint is loaded directly.
            plot: If True, saves score distribution, ROC/PR, and confusion
                matrix plots to ``save_path/plots/``.

        Returns:
            Dictionary of MIA metrics including AUC, accuracy, precision,
            recall, F1, score statistics, and Wasserstein distances.
        """
        print("Starting FlowMIA-GAN privacy evaluation...")

        attack, result = self._run_attack(
            "flowmia_gan",
            fit_kwargs={
                "epochs": self.config["num_epochs"],
                "batch_size": self.config["batch_size"],
                "fcheckpoint": self.config["fcheckpoint"],
                "save_path": self.save_path,
                "pre_trained_model": pre_trained_model,
                "use_wgan": self.use_wgan,
            },
            attack_kwargs={"test_size": test_size},
        )
        self.flowmia_gan = attack._gan
        self.mia_results = result.extra
        print(f"FlowMIA-GAN results: AUC={self.mia_results['auc']:.4f}")

        if plot:
            attack.plot_all(result, save_path=self.save_path)

        return self.mia_results

    def domias(self, test_size: int = 1000, epochs: int = 10, save_path: str = "", load: bool = False) -> tuple[np.ndarray, float]:
        """
        Run the DOMIAS membership inference attack.

        Estimates the log-density of the synthetic distribution at each
        query point using a Block Neural Autoregressive Flow (BNAF). A
        higher density score implies the sample is more likely a member.

        Args:
            test_size: Number of members and non-members to sample for
                evaluation (balanced split).

        Returns:
            Tuple of (scores, auc) where ``scores`` is a 1-D array of
            log-density values and ``auc`` is the ROC-AUC of the attack.
        """
        _, result = self._run_attack(
            "domias",
            attack_kwargs={
                "test_size": test_size,
                "epochs": epochs,
                "save_path": save_path,
                "load": load,
            },
        )
        print(f"DOMIAS results: AUC={result.auc:.4f}")
        return result.scores, result.auc

    def compute_dcr(self, test_size: int = 1000) -> tuple[np.ndarray, float]:
        """
        Run the DCR (Distance to Closest Record) membership inference attack.

        Assigns each query point a score equal to the negative distance to
        its nearest neighbor in the synthetic dataset. Members are expected
        to have smaller distances (higher scores) than non-members.

        Args:
            test_size: Number of members and non-members to sample for
                evaluation (balanced split).

        Returns:
            Tuple of (scores, auc) where ``scores`` is a 1-D array of
            negative-distance values and ``auc`` is the ROC-AUC of the attack.
        """
        _, result = self._run_attack("dcr", attack_kwargs={"test_size": test_size})
        print(f"DCR results: AUC={result.auc:.4f}")
        return result.scores, result.auc

    def evaluate_privacy(
        self,
        attacks: list = None,
        plot: bool = False,
        run_domias: bool = False,
        run_dcr: bool = False,
        test_size: int = 1000,
        attack_kwargs: dict = None,
    ) -> dict:
        """
        Run privacy attacks and collect results.

        Args:
            attacks: Names of registered attacks to run (see
                :func:`flowmia.attacks.list_attacks`). Defaults to
                ``["flowmia_gan"]`` plus ``"domias"``/``"dcr"`` depending on
                ``run_domias``/``run_dcr``, matching the legacy behavior.
                Pass this explicitly to run any registered attack, including
                custom ones added via :func:`flowmia.attacks.register_attack`.
            plot: Forward to :meth:`flowmiagan` to save diagnostic plots.
            run_domias: Whether to include the DOMIAS attack (ignored if
                ``attacks`` is given explicitly).
            run_dcr: Whether to include the DCR attack (ignored if ``attacks``
                is given explicitly).
            test_size: Evaluation sample size, used as the default
                ``attack_kwargs[name]["test_size"]`` for any attack not
                otherwise configured.
            attack_kwargs: Optional ``{attack_name: {"fit": {...}, "attack": {...}}}``
                overrides for attack-specific hyperparameters.

        Returns:
            Dictionary keyed by attack name. Each value is an
            :class:`~flowmia.attacks.AttackResult`, except ``"flowmia_gan"``
            which is a plain dict (legacy shape) when using the default
            ``attacks`` list, for backward compatibility with existing code.
        """
        legacy_mode = attacks is None
        if legacy_mode:
            attacks = ["flowmia_gan"]
            if run_domias:
                attacks.append("domias")
            if run_dcr:
                attacks.append("dcr")
        attack_kwargs = attack_kwargs or {}

        results = {}
        for name in attacks:
            if legacy_mode and name == "flowmia_gan":
                results[name] = self.flowmiagan(plot=plot)
                continue

            kwargs = attack_kwargs.get(name, {})
            fit_kw = dict(kwargs.get("fit", {}))
            attack_kw = dict(kwargs.get("attack", {"test_size": test_size}))
            attack, result = self._run_attack(name, fit_kwargs=fit_kw, attack_kwargs=attack_kw)
            print(f"{name} results: AUC={result.auc:.4f}")
            if plot and hasattr(attack, "plot_all"):
                attack.plot_all(result, save_path=self.save_path)
            results[name] = result

        if legacy_mode:
            results.setdefault("domias", None)
            results.setdefault("dcr", None)

        return results
    
    # ------------------------------------------------------------------
    # Fidelity
    # ------------------------------------------------------------------
 
    def evaluate_fidelity(self, plot: bool = False) -> dict:
        """
        Measure per-feature distributional similarity between real and synthetic data.
 
        Computes KL divergence, Jensen-Shannon divergence, and Wasserstein distance
        for every column. Categorical columns use probability mass functions over
        observed categories; numerical columns use histogram-based estimates.
 
        Args:
            plot: If True, saves a grouped bar chart (log scale) of all three
                divergences per feature to ``save_path/plots/fidelity.pdf``.
 
        Returns:
            Dictionary mapping each column name to a (1, 3) numpy array
            containing ``[kl, js, wasserstein]`` divergence values.
        """
        self.fidelity_results = fidelity_compute(
            self.member,
            self.synth,
            categorical=self.categorical_cols + [self.label_col],
        )
 
        if plot:
            plotFidelity(self.fidelity_results, self.save_path)
 
        return self.fidelity_results
 
    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
 
    def evaluate_utility(self, classifiers: list, plot: bool = False) -> dict:
        """
        Benchmark synthetic data utility via RTR and TSTR protocols.
 
        For each classifier in ``classifiers``, two pipelines are evaluated:
 
        - **RTR** (Real-Train / Real-Test): trained on real member data, tested
          on the held-out test set. Serves as an upper-bound reference.
        - **TSTR** (Synthetic-Train / Real-Test): trained on synthetic data,
          tested on the same held-out test set. A TSTR score close to RTR
          indicates high utility.
 
        Each pipeline includes its own RobustScaler + OneHotEncoder fit, so
        no pre-transformed arrays are needed here.
 
        Args:
            classifiers: List of scikit-learn compatible classifier instances.
            plot: If True, saves a 2×2 grid of metric bar charts to
                ``save_path/plots/utility.pdf``.
 
        Returns:
            Nested dictionary of the form
            ``{classifier_name: {"RTR": metrics_dict, "TSTR": metrics_dict}}``
            where each ``metrics_dict`` contains Accuracy, Precision, Recall,
            and F1-Score.
        """
        utility = Utility(
            classifiers,
            self.member,
            self.util_test,
            self.synth,
            self.categorical_cols,
            self.numerical_cols,
            self.ip_cols,
            self.label_col,
        )
        utility_dict = utility.evaluate()
 
        if plot:
            utility.plot_utility(utility_dict, self.save_path)
 
        return utility_dict
 
    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------
 
    def evaluate_all(
        self,
        classifiers: list,
        plot: bool = False,
        run_domias: bool = False,
        run_dcr: bool = False,
        test_size: int = 1000,
    ) -> dict:
        """
        Run the complete evaluation pipeline: privacy, fidelity, and utility.
 
        This is a convenience method that calls :meth:`evaluate_privacy`,
        :meth:`evaluate_fidelity`, and :meth:`evaluate_utility` in sequence
        and bundles their outputs into a single dictionary.
 
        Args:
            classifiers: List of scikit-learn classifiers for utility evaluation.
            plot: If True, all diagnostic plots are saved under ``save_path/plots/``.
            run_domias: Whether to include the DOMIAS attack in privacy evaluation.
            run_dcr: Whether to include the DCR attack in privacy evaluation.
            test_size: Evaluation sample size for DOMIAS and DCR.
 
        Returns:
            Dictionary with keys ``"privacy"``, ``"fidelity"``, and ``"utility"``,
            each containing the results of the corresponding evaluation.
        """
        return {
            "privacy": self.evaluate_privacy(
                plot=plot,
                run_domias=run_domias,
                run_dcr=run_dcr,
                test_size=test_size,
            ),
            "fidelity": self.evaluate_fidelity(plot=plot),
            "utility": self.evaluate_utility(classifiers, plot=plot),
        }