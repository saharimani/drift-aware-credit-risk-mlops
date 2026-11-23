# Champion state, scoring and champion-specific delayed evidence.
# Copied from Notebook 05 cells 42-53, with the champion feature list, thresholds
# and the cohort's own immediate signals passed in instead of read from globals.

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

from signals import calculate_jsd_from_counts


def to_log_odds(probabilities):
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def score_raw(champion_state, feature_df, champion_feature_cols):
    return champion_state["underlying_model"].predict_proba(
        feature_df[champion_feature_cols])[:, 1]


def apply_recalibration_map(recalibration_map, raw_scores):
    return recalibration_map.predict_proba(to_log_odds(raw_scores).reshape(-1, 1))[:, 1]


def score_champion(champion_state, feature_df, champion_feature_cols):
    raw_scores = score_raw(champion_state, feature_df, champion_feature_cols)

    if champion_state["active_recalibration_map"] is None:
        return raw_scores

    return apply_recalibration_map(champion_state["active_recalibration_map"], raw_scores)


def fit_recalibration_map(raw_scores, outcomes):
    recalibration_map = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    recalibration_map.fit(to_log_odds(raw_scores).reshape(-1, 1), outcomes)
    return recalibration_map


def fit_calibration_diagnostic(deployed_scores, outcomes):
    diagnostic = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    diagnostic.fit(to_log_odds(deployed_scores).reshape(-1, 1), outcomes)
    return float(diagnostic.intercept_[0]), float(diagnostic.coef_[0][0])


def evaluate_predictions(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    actual_bad_loan_rate = y_true.mean()
    mean_predicted_probability = y_score.mean()
    two_classes = pd.Series(y_true).nunique() > 1

    return {"rows": len(y_true),
            "actual_bad_loan_rate": actual_bad_loan_rate,
            "mean_predicted_probability": mean_predicted_probability,
            "calibration_gap": actual_bad_loan_rate - mean_predicted_probability,
            "absolute_calibration_gap": abs(actual_bad_loan_rate - mean_predicted_probability),
            "brier_score": brier_score_loss(y_true, y_score),
            "log_loss": log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6), labels=[0, 1]),
            "roc_auc": roc_auc_score(y_true, y_score) if two_classes else np.nan,
            "average_precision": average_precision_score(y_true, y_score) if two_classes else np.nan}


def assign_score_bins(scores, edges):
    scores = pd.Series(scores).astype(float)
    internal_edges = np.asarray(edges[1:-1], dtype=float)
    return np.searchsorted(internal_edges, scores, side="right")


def build_champion_reference(reference_outcomes, deployed_scores, label,
                             edges=None, baseline_metrics=None):
    scores = np.asarray(deployed_scores, dtype=float)
    outcomes = np.asarray(reference_outcomes)

    if edges is None:
        edges = np.unique(np.quantile(scores, np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf

    counts, _ = np.histogram(scores, bins=edges)

    bin_table = (pd.DataFrame({"score_bin": assign_score_bins(scores, edges),
                               "y": outcomes})
                 .groupby("score_bin")
                 .agg(baseline_rows=("y", "size"),
                      baseline_bad_loan_rate=("y", "mean"))
                 .reset_index())

    intercept, slope = fit_calibration_diagnostic(scores, outcomes)

    if baseline_metrics is None:
        baseline_metrics = evaluate_predictions(outcomes, scores)

    return {"label": label,
            "score_edges": edges,
            "reference_counts": counts,
            "score_bin_baseline": bin_table,
            "baseline_metrics": baseline_metrics,
            "reference_calibration_intercept": intercept,
            "reference_calibration_slope": slope}


def build_starting_champion(champion_model, concept_drift_baseline,
                            calibration_scored_df, score_edges, cfg):
    """The original Notebook 03 champion, rebuilt exactly as in Notebook 05 cell 52."""
    baseline_row = concept_drift_baseline.iloc[0]

    baseline_metrics = {"rows": int(calibration_scored_df.shape[0]),
                        "actual_bad_loan_rate": float(baseline_row["actual_bad_loan_rate"]),
                        "mean_predicted_probability": float(baseline_row["mean_predicted_probability"]),
                        "calibration_gap": float(baseline_row["calibration_gap"]),
                        "absolute_calibration_gap": abs(float(baseline_row["calibration_gap"])),
                        "brier_score": float(baseline_row["brier_score"]),
                        "log_loss": float(baseline_row["log_loss"]),
                        "roc_auc": float(baseline_row["roc_auc"]),
                        "average_precision": float(baseline_row["average_precision"])}

    reference = build_champion_reference(
        calibration_scored_df[cfg["target_col"]].values,
        calibration_scored_df[cfg["prediction_col"]].values,
        cfg["starting_reference_label"],
        edges=score_edges,
        baseline_metrics=baseline_metrics)

    return {"champion_id": cfg["starting_champion_id"],
            "underlying_model": champion_model,
            "active_recalibration_map": None,
            "deployment_month": "start",
            "action_type": "original",
            "reference": reference}


def compute_champion_evidence(monitor_month, y_month, deployed_scores, reference, art, cfg):
    """Delayed evidence for one mature cohort, against its own champion's reference."""
    metrics = evaluate_predictions(y_month, deployed_scores)
    base = reference["baseline_metrics"]

    brier_change = metrics["brier_score"] - base["brier_score"]
    log_loss_change = metrics["log_loss"] - base["log_loss"]
    auc_drop = base["roc_auc"] - metrics["roc_auc"]
    average_precision_change = metrics["average_precision"] - base["average_precision"]

    performance_high = (metrics["absolute_calibration_gap"] >= art["calibration_gap_high_threshold"] or
                        brier_change >= art["brier_high_threshold"] or
                        auc_drop >= art["auc_drop_high_threshold"])

    performance_moderate = (metrics["absolute_calibration_gap"] >= art["calibration_gap_moderate_threshold"] or
                            brier_change >= art["brier_moderate_threshold"] or
                            auc_drop >= art["auc_drop_moderate_threshold"])

    performance_level = "high" if performance_high else ("moderate" if performance_moderate else "low")

    bins = assign_score_bins(deployed_scores, reference["score_edges"])

    current = (pd.DataFrame({"score_bin": bins, "y": np.asarray(y_month)})
               .groupby("score_bin")
               .agg(current_rows=("y", "size"),
                    current_bad_loan_rate=("y", "mean"))
               .reset_index())

    comparison = reference["score_bin_baseline"].merge(current, on="score_bin", how="left")
    comparison["current_rows"] = comparison["current_rows"].fillna(0)
    valid = comparison[comparison["current_rows"] >= cfg["concept_min_bin_rows"]].copy()

    if valid.shape[0] == 0:
        mean_abs_shift = signed_shift = residual_mean_abs = residual_max_abs = np.nan
    else:
        valid["bin_change"] = valid["current_bad_loan_rate"] - valid["baseline_bad_loan_rate"]
        weights = valid["current_rows"] / valid["current_rows"].sum()
        mean_abs_shift = float(np.sum(weights * valid["bin_change"].abs()))
        signed_shift = float(np.sum(weights * valid["bin_change"]))
        residual = valid["bin_change"] - signed_shift
        residual_mean_abs = float(np.sum(weights * residual.abs()))
        residual_max_abs = float(residual.abs().max())

    localisation = mean_abs_shift - abs(signed_shift)

    if pd.isna(mean_abs_shift) or mean_abs_shift < art["concept_moderate_threshold"]:
        concept_level = "low"
    elif mean_abs_shift >= art["concept_high_threshold"]:
        concept_level = "high"
    else:
        concept_level = "moderate"

    if concept_level == "low":
        shift_type = "low_or_no_material_shift"
    elif pd.notna(residual_mean_abs) and residual_mean_abs >= art["fork"]["residual_bin_shift_threshold"]:
        shift_type = "localised_or_nonparallel_shift"
    else:
        shift_type = "largely_global_level_shift"

    intercept, slope = fit_calibration_diagnostic(deployed_scores, np.asarray(y_month))

    return {"monitor_month": monitor_month,
            **metrics,
            "brier_change": brier_change,
            "log_loss_change": log_loss_change,
            "auc_drop": auc_drop,
            "average_precision_change": average_precision_change,
            "performance_level": performance_level,
            "score_bin_mean_absolute_shift": mean_abs_shift,
            "score_bin_signed_shift": signed_shift,
            "residual_mean_abs_shift": residual_mean_abs,
            "residual_max_abs_shift": residual_max_abs,
            "concept_localisation": localisation,
            "concept_level": concept_level,
            "concept_shift_type": shift_type,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "reference_calibration_intercept": reference["reference_calibration_intercept"],
            "reference_calibration_slope": reference["reference_calibration_slope"],
            "calibration_slope_deviation": slope - reference["reference_calibration_slope"]}


def target_drift_for_cohort(y_month, art):
    """Portfolio target drift against the 2013 calibration baseline (Notebook 04 section 5)."""
    baseline_rate = art["baseline_bad_loan_rate"]
    change = float(np.asarray(y_month).mean() - baseline_rate)

    moderate = float(art["monitoring_metadata"]["target_moderate_threshold"])
    high = float(art["monitoring_metadata"]["target_high_threshold"])

    level = "high" if change >= high else ("moderate" if change >= moderate else "low")

    return {"target_rate_change": change,
            "target_drift_level": level,
            "target_drift_trigger": bool(change >= moderate)}
