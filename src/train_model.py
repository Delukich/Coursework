import os
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import CATEGORICAL_COLUMNS_USED, FEATURE_META_PATH, MODEL_PATH

warnings.filterwarnings("ignore")

plt.style.use("ggplot")
sns.set_palette("viridis")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ProfitabilityClassifier:
    DEFAULT_XGB_PARAMS = {
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "enable_categorical": True,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }

    def __init__(self):
        self.feature_names: list[str] = []
        self.medians: dict[str, float] = {}
        self.best_model = None
        self.optimal_threshold: float = 0.5

    def _cast_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        for col in CATEGORICAL_COLUMNS_USED:
            if col in prepared.columns:
                prepared[col] = prepared[col].astype("category")
        return prepared

    def _drop_remaining_object_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        obj_cols = prepared.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            print(f"  Dropped unsupported object columns: {obj_cols}")
            prepared = prepared.drop(columns=obj_cols)
        return prepared

    def _fill_numeric_missing(
        self,
        frame: pd.DataFrame,
        medians: pd.Series | dict[str, float],
    ) -> pd.DataFrame:
        prepared = frame.copy()
        num_cols = prepared.select_dtypes(include=np.number).columns
        medians_s = medians if isinstance(medians, pd.Series) else pd.Series(medians)
        if len(num_cols) > 0:
            prepared[num_cols] = prepared[num_cols].fillna(medians_s)
        return prepared

    def prepare_datasets(self, df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
        print("\n[1/5] Підготовка даних для XGBoost...")

        if "Is_Profitable" not in df.columns:
            raise ValueError("Column 'Is_Profitable' is missing. Run the data pipeline first.")

        prepared = self._cast_categorical_columns(df)
        prepared = self._drop_remaining_object_columns(prepared)

        X = prepared.drop(columns=["Is_Profitable"], errors="ignore")
        y = prepared["Is_Profitable"]

        balance = y.value_counts(normalize=True)
        print(
            f"  Баланс класів: прибуткові={balance.get(1, 0):.1%} | "
            f"збиткові={balance.get(0, 0):.1%}"
        )

        (
            X_train,
            X_val,
            X_test,
            X_dev,
            y_train,
            y_val,
            y_test,
            y_dev,
        ) = self._split_train_val_test(X, y)
        self.feature_names = X_train.columns.tolist()
        self.medians = X_train.select_dtypes(include=np.number).median().to_dict()

        X_train = self._fill_numeric_missing(X_train, self.medians)
        X_val = self._fill_numeric_missing(X_val, self.medians)
        X_test = self._fill_numeric_missing(X_test, self.medians)
        X_dev = self._fill_numeric_missing(X_dev, self.medians)

        print(
            f"  Train: {len(X_train):,} | Validation: {len(X_val):,} | Test: {len(X_test):,}"
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "X_dev": X_dev,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "y_dev": y_dev,
        }

    def _split_train_val_test(self, X: pd.DataFrame, y: pd.Series):
        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev, test_size=0.1765, random_state=42, stratify=y_dev
        )
        return X_train, X_val, X_test, X_dev, y_train, y_val, y_test, y_dev

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, params: dict | None = None) -> dict:
        print("\n[2/5] Запуск StratifiedKFold CV (5 фолдів)...")

        cv_params = self.DEFAULT_XGB_PARAMS.copy()
        if params:
            cv_params.update(params)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = {"auc": [], "acc": [], "f1": []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_fold_train = X.iloc[train_idx].copy()
            X_fold_val = X.iloc[val_idx].copy()
            y_fold_train = y.iloc[train_idx]
            y_fold_val = y.iloc[val_idx]

            fold_medians = X_fold_train.select_dtypes(include=np.number).median()
            X_fold_train = self._fill_numeric_missing(X_fold_train, fold_medians)
            X_fold_val = self._fill_numeric_missing(X_fold_val, fold_medians)

            model = xgb.XGBClassifier(**cv_params)
            model.fit(X_fold_train, y_fold_train, verbose=False)

            proba = model.predict_proba(X_fold_val)[:, 1]
            preds = (proba >= 0.5).astype(int)

            auc = roc_auc_score(y_fold_val, proba)
            acc = accuracy_score(y_fold_val, preds)
            f1 = f1_score(y_fold_val, preds, average="macro", zero_division=0)

            fold_metrics["auc"].append(auc)
            fold_metrics["acc"].append(acc)
            fold_metrics["f1"].append(f1)
            print(f"  Fold {fold}: AUC={auc:.4f} | Acc={acc:.4f} | F1={f1:.4f}")

        print("\n  Підсумок CV:")
        for metric, values in fold_metrics.items():
            print(f"    {metric.upper()}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

        return {
            name: {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
            for name, values in fold_metrics.items()
        }

    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        if not OPTUNA_AVAILABLE:
            print("\n[3/5] Optuna is not installed. Using default XGBoost parameters.")
            return self.DEFAULT_XGB_PARAMS.copy()

        print("\n[3/5] Пошук гіперпараметрів через Optuna...")

        def objective(trial):
            params = {
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",
                "enable_categorical": True,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        study = optuna.create_study(direction="maximize", study_name="profitability_classifier")
        study.optimize(objective, n_trials=50, timeout=300, show_progress_bar=False)

        print(
            f"  Найкращий validation ROC-AUC: {study.best_value:.4f} "
            f"(trial #{study.best_trial.number})"
        )

        best_params = self.DEFAULT_XGB_PARAMS.copy()
        best_params.update(study.best_params)
        return best_params

    def choose_threshold(self, y_true: pd.Series, preds_proba: np.ndarray) -> float:
        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in np.arange(0.05, 0.95, 0.01):
            preds = (preds_proba >= threshold).astype(int)
            score = f1_score(y_true, preds, average="macro", zero_division=0)
            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(threshold)

        print(
            f"  Обраний поріг за validation split: "
            f"{best_threshold:.2f} (macro F1={best_f1:.4f})"
        )
        return best_threshold

    def train_evaluate(self, df: pd.DataFrame):
        datasets = self.prepare_datasets(df)
        X_train, X_val, X_test, X_dev = (
            datasets["X_train"],
            datasets["X_val"],
            datasets["X_test"],
            datasets["X_dev"],
        )
        y_train, y_val, y_test, y_dev = (
            datasets["y_train"],
            datasets["y_val"],
            datasets["y_test"],
            datasets["y_dev"],
        )
        cv_results = self.cross_validate(X_dev, y_dev)
        best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        val_proba = self._train_for_threshold(X_train, y_train, X_val, y_val, best_params)
        test_proba, final_preds = self._train_final_model(X_dev, y_dev, X_test, y_test, best_params)
        acc, balanced_acc, roc, ap, mcc, kappa, brier = self._score_metrics(
            y_test, final_preds, test_proba
        )

        print(f"\n{'=' * 58}")
        print(f"  Accuracy         : {acc:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  ROC AUC          : {roc:.4f}")
        print(f"  Average Precision: {ap:.4f}")
        print(f"  MCC              : {mcc:.4f}")
        print(f"  Cohen Kappa      : {kappa:.4f}")
        print(f"  Brier Score      : {brier:.4f}")
        print(f"  Optimal Threshold: {self.optimal_threshold:.2f}")
        print(f"{'=' * 58}")
        print(
            classification_report(
                y_test,
                final_preds,
                target_names=["Not Profitable (0)", "Profitable (1)"],
            )
        )

        os.makedirs("plots", exist_ok=True)
        self._plot_confusion_matrix(y_test, final_preds)
        self._plot_roc_curve(y_test, test_proba, roc)
        self._plot_precision_recall(y_test, test_proba, ap)
        self._plot_feature_importance()
        self._plot_threshold_search(y_val, val_proba)
        self._plot_calibration_curve(y_test, test_proba)
        self._plot_probability_distribution(y_test, test_proba)
        if SHAP_AVAILABLE:
            self._plot_shap(X_test)
        else:
            print("  SHAP is not installed. Skipping SHAP plots.")

        metrics_summary = {
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "roc_auc": roc,
            "avg_precision": ap,
            "mcc": mcc,
            "cohen_kappa": kappa,
            "brier_score": brier,
        }
        self._save_metrics_report(metrics_summary, cv_results, y_test, final_preds)
        self._save_artifacts(metrics_summary, cv_results)

    def _train_for_threshold(self, X_train, y_train, X_val, y_val, best_params):
        print("\n[4/5] Навчання фінальної моделі на train + validation...")
        validation_model = xgb.XGBClassifier(**best_params)
        validation_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_proba = validation_model.predict_proba(X_val)[:, 1]
        self.optimal_threshold = self.choose_threshold(y_val, val_proba)
        return val_proba

    def _train_final_model(self, X_dev, y_dev, X_test, y_test, best_params):
        self.best_model = xgb.XGBClassifier(**best_params)
        self.best_model.fit(X_dev, y_dev, eval_set=[(X_test, y_test)], verbose=False)
        print("\n[5/5] Оцінка на відкладеній тестовій вибірці...")
        test_proba = self.best_model.predict_proba(X_test)[:, 1]
        final_preds = (test_proba >= self.optimal_threshold).astype(int)
        return test_proba, final_preds

    def _score_metrics(self, y_test, final_preds, test_proba):
        acc = accuracy_score(y_test, final_preds)
        balanced_acc = balanced_accuracy_score(y_test, final_preds)
        roc = roc_auc_score(y_test, test_proba)
        ap = average_precision_score(y_test, test_proba)
        mcc = matthews_corrcoef(y_test, final_preds)
        kappa = cohen_kappa_score(y_test, final_preds)
        brier = brier_score_loss(y_test, test_proba)
        return acc, balanced_acc, roc, ap, mcc, kappa, brier

    def _save_artifacts(self, metrics_summary: dict, cv_results: dict):
        joblib.dump(self.best_model, MODEL_PATH)
        joblib.dump(
            {
                "feature_names": self.feature_names,
                "medians": self.medians,
                "optimal_threshold": self.optimal_threshold,
                "roc_auc": metrics_summary["roc_auc"],
                "avg_precision": metrics_summary["avg_precision"],
                "accuracy": metrics_summary["accuracy"],
                "balanced_accuracy": metrics_summary["balanced_accuracy"],
                "mcc": metrics_summary["mcc"],
                "cohen_kappa": metrics_summary["cohen_kappa"],
                "brier_score": metrics_summary["brier_score"],
                "cv_auc_mean": cv_results["auc"]["mean"],
                "cv_auc_std": cv_results["auc"]["std"],
            },
            FEATURE_META_PATH,
        )
        print(f"\nModel saved to: {MODEL_PATH}")
        print(f"Metadata saved to: {FEATURE_META_PATH}")

    def _plot_confusion_matrix(self, y_true, preds):
        cm = confusion_matrix(y_true, preds)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Збиткове", "Прибуткове"],
            yticklabels=["Збиткове", "Прибуткове"],
            ax=ax,
        )
        ax.set_title(f"Матриця помилок (поріг={self.optimal_threshold:.2f})")
        ax.set_ylabel("Фактичний клас")
        ax.set_xlabel("Прогнозований клас")
        fig.tight_layout()
        fig.savefig("plots/confusion_matrix.png", dpi=150)
        plt.close(fig)
        print("  Saved: plots/confusion_matrix.png")

    def _plot_roc_curve(self, y_true, preds_proba, roc_score):
        fpr, tpr, _ = roc_curve(y_true, preds_proba)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="navy", lw=2, label=f"ROC AUC = {roc_score:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
        ax.set_xlabel("Частка хибнопозитивних")
        ax.set_ylabel("Частка істиннопозитивних")
        ax.set_title("ROC-крива")
        ax.legend()
        fig.tight_layout()
        fig.savefig("plots/roc_curve.png", dpi=150)
        plt.close(fig)
        print("  Saved: plots/roc_curve.png")

    def _plot_precision_recall(self, y_true, preds_proba, ap_score):
        precision, recall, _ = precision_recall_curve(y_true, preds_proba)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, color="darkorange", lw=2, label=f"Середня precision = {ap_score:.3f}")
        ax.set_xlabel("Повнота (Recall)")
        ax.set_ylabel("Точність (Precision)")
        ax.set_title("Крива Precision-Recall")
        ax.legend()
        fig.tight_layout()
        fig.savefig("plots/precision_recall.png", dpi=150)
        plt.close(fig)
        print("  Saved: plots/precision_recall.png")

    def _plot_feature_importance(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        for ax, importance_type in zip(axes, ["gain", "weight"]):
            xgb.plot_importance(
                self.best_model,
                importance_type=importance_type,
                max_num_features=15,
                ax=ax,
                title=f"Важливість ознак ({importance_type})",
            )
        fig.tight_layout()
        fig.savefig("plots/feature_importance.png", dpi=150)
        plt.close(fig)
        print("  Saved: plots/feature_importance.png")

    def _plot_threshold_search(self, y_true, preds_proba):
        thresholds = np.arange(0.05, 0.95, 0.01)
        f1_scores, precisions, recalls = [], [], []

        for threshold in thresholds:
            preds = (preds_proba >= threshold).astype(int)
            f1_scores.append(f1_score(y_true, preds, average="macro", zero_division=0))
            precisions.append(precision_score(y_true, preds, average="macro", zero_division=0))
            recalls.append(recall_score(y_true, preds, average="macro", zero_division=0))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thresholds, f1_scores, label="F1 (macro)", lw=2)
        ax.plot(thresholds, precisions, label="Точність", lw=1.5, linestyle="--")
        ax.plot(thresholds, recalls, label="Повнота", lw=1.5, linestyle=":")
        ax.axvline(
            self.optimal_threshold,
            color="red",
            linestyle="-",
            lw=1.5,
            label=f"Оптимум={self.optimal_threshold:.2f}",
        )
        ax.set_xlabel("Поріг класифікації")
        ax.set_ylabel("Значення метрики")
        ax.set_title("Метрики на валідації залежно від порогу")
        ax.legend()
        fig.tight_layout()
        fig.savefig("plots/threshold_search.png", dpi=150)
        plt.close(fig)
        print("  Saved: plots/threshold_search.png")

    def _plot_calibration_curve(self, y_true, preds_proba):
        frac_pos, mean_pred = calibration_curve(y_true, preds_proba, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(mean_pred, frac_pos, marker="o", lw=2, label="Модель")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ідеальна калібровка")
        ax.set_xlabel("Середня передбачена ймовірність")
        ax.set_ylabel("Фактична частка позитивного класу")
        ax.set_title("Крива калібрування")
        ax.legend()
        fig.tight_layout()
        fig.savefig("plots/calibration_curve.png", dpi=150)
        plt.close(fig)
        print("  Saved: plots/calibration_curve.png")

    def _plot_probability_distribution(self, y_true, preds_proba):
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(
            preds_proba[y_true == 0],
            color="crimson",
            label="Збиткове",
            stat="density",
            bins=30,
            alpha=0.45,
            ax=ax,
        )
        sns.histplot(
            preds_proba[y_true == 1],
            color="seagreen",
            label="Прибуткове",
            stat="density",
            bins=30,
            alpha=0.45,
            ax=ax,
        )
        ax.axvline(
            self.optimal_threshold,
            color="black",
            linestyle="--",
            lw=1.5,
            label=f"Поріг={self.optimal_threshold:.2f}",
        )
        ax.set_xlabel("Передбачена ймовірність")
        ax.set_ylabel("Щільність")
        ax.set_title("Розподіл передбачених ймовірностей")
        ax.legend()
        fig.tight_layout()
        fig.savefig("plots/probability_distribution.png", dpi=150)
        plt.close(fig)
        print("  Saved: plots/probability_distribution.png")

    def _plot_shap(self, X_test: pd.DataFrame):
        try:
            print("  Calculating SHAP values...")
            X_sample = X_test.head(500)
            explainer = shap.TreeExplainer(self.best_model)
            shap_values = explainer.shap_values(X_sample)

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig("plots/shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  Saved: plots/shap_summary.png")

            plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
            plt.tight_layout()
            plt.savefig("plots/shap_bar.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  Saved: plots/shap_bar.png")
        except Exception as exc:
            print(f"  SHAP failed: {exc}")

    def _save_metrics_report(self, metrics_summary: dict, cv_results, y_true, final_preds):
        report_path = "plots/metrics_report.txt"
        lines = [
            "=" * 58,
            "Звіт про метрики класифікатора прибутковості",
            "=" * 58,
            "",
            f"Точність (Accuracy)           : {metrics_summary['accuracy']:.4f}",
            f"Збалансована точність         : {metrics_summary['balanced_accuracy']:.4f}",
            f"ROC AUC                       : {metrics_summary['roc_auc']:.4f}",
            f"Середня precision             : {metrics_summary['avg_precision']:.4f}",
            f"MCC                           : {metrics_summary['mcc']:.4f}",
            f"Коефіцієнт Каппа Коена        : {metrics_summary['cohen_kappa']:.4f}",
            f"Brier Score                   : {metrics_summary['brier_score']:.4f}",
            f"Оптимальний поріг             : {self.optimal_threshold:.2f}",
            "",
            "Крос-валідація (5 фолдів на development-вибірці):",
            f"  AUC : {cv_results['auc']['mean']:.4f} +/- {cv_results['auc']['std']:.4f}",
            f"  Acc : {cv_results['acc']['mean']:.4f} +/- {cv_results['acc']['std']:.4f}",
            f"  F1  : {cv_results['f1']['mean']:.4f} +/- {cv_results['f1']['std']:.4f}",
            "",
            "Класифікаційний звіт:",
            classification_report(
                y_true,
                final_preds,
                target_names=["Збиткове (0)", "Прибуткове (1)"],
            ),
        ]
        with open(report_path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines))
        print(f"  Saved: {report_path}")


if __name__ == "__main__":
    from src.config import ENRICHED_OUTPUT_PATH

    if os.path.exists(ENRICHED_OUTPUT_PATH):
        ProfitabilityClassifier().train_evaluate(pd.read_csv(ENRICHED_OUTPUT_PATH, low_memory=False))
    else:
        print(f"File not found: {ENRICHED_OUTPUT_PATH}\nRun the data pipeline first.")
