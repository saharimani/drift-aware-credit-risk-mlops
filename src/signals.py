# Immediate monitoring signals.
# Copied unchanged from Notebook 04 cells 16-22, 38-41 and 100-102, with the
# 2013 reference frame passed in as an argument rather than read from a global.

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def calculate_psi_from_counts(reference_counts, current_counts, epsilon=1e-6):
    reference = np.asarray(reference_counts, dtype=float)
    current = np.asarray(current_counts, dtype=float)

    if reference.sum() == 0 or current.sum() == 0:
        return np.nan

    reference = reference / reference.sum()
    current = current / current.sum()

    reference = np.where(reference == 0, epsilon, reference)
    current = np.where(current == 0, epsilon, current)

    reference = reference / reference.sum()
    current = current / current.sum()

    return np.sum((current - reference) * np.log(current / reference))


def classify_psi(psi_value):
    if pd.isna(psi_value):
        return "missing"
    elif psi_value >= 0.25:
        return "high"
    elif psi_value >= 0.10:
        return "moderate"
    else:
        return "low"


def chi_square_drift_p_value(reference_counts, current_counts):
    table = np.vstack([np.asarray(reference_counts, dtype=float),
                       np.asarray(current_counts, dtype=float)])

    keep_cols = table.sum(axis=0) > 0
    table = table[:, keep_cols]

    if table.shape[1] < 2:
        return np.nan

    if table[0].sum() == 0 or table[1].sum() == 0:
        return np.nan

    try:
        _, p_value, _, _ = chi2_contingency(table, correction=False)
        return p_value
    except ValueError:
        return np.nan


def benjamini_hochberg_adjustment(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan, dtype=float)

    valid_mask = ~np.isnan(p_values)
    valid_p = p_values[valid_mask]

    if len(valid_p) == 0:
        return adjusted

    n_tests = len(valid_p)
    order = np.argsort(valid_p)
    ranked_p = valid_p[order]

    adjusted_ranked = ranked_p * n_tests / np.arange(1, n_tests + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)

    adjusted_valid = np.empty(n_tests)
    adjusted_valid[order] = adjusted_ranked

    adjusted[valid_mask] = adjusted_valid
    return adjusted


def prepare_discrete_values(series):
    return (series.astype("object")
            .where(series.notna(), "__MISSING__")
            .astype(str))


def numeric_feature_drift(reference_df, current_df, feature, bin_edges):
    reference_values = pd.to_numeric(reference_df[feature], errors="coerce").dropna()
    current_values = pd.to_numeric(current_df[feature], errors="coerce").dropna()

    if len(reference_values) == 0 or len(current_values) == 0:
        return {"psi": np.nan, "p_value": np.nan}

    safe_edges = np.asarray(bin_edges, dtype=float).copy()
    safe_edges[0] = -np.inf
    safe_edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference_values, bins=safe_edges)
    current_counts, _ = np.histogram(current_values, bins=safe_edges)

    return {"psi": calculate_psi_from_counts(reference_counts, current_counts),
            "p_value": chi_square_drift_p_value(reference_counts, current_counts)}


def discrete_feature_drift(reference_df, current_df, feature):
    reference_values = prepare_discrete_values(reference_df[feature])
    current_values = prepare_discrete_values(current_df[feature])

    reference_counts_series = reference_values.value_counts(dropna=False)
    current_counts_series = current_values.value_counts(dropna=False)

    all_categories = sorted(set(reference_counts_series.index).union(
                            set(current_counts_series.index)))

    reference_counts = reference_counts_series.reindex(all_categories, fill_value=0).values
    current_counts = current_counts_series.reindex(all_categories, fill_value=0).values

    return {"psi": calculate_psi_from_counts(reference_counts, current_counts),
            "p_value": chi_square_drift_p_value(reference_counts, current_counts)}


def feature_drift_for_month(reference_df, cohort_df, art):
    """One month of feature drift, then the Notebook 04 governance trigger.

    The Benjamini-Hochberg adjustment is applied within the month across every
    monitored feature, exactly as in the notebook.
    """
    rows = []

    for feature in art["numeric_psi_cols"]:
        result = numeric_feature_drift(reference_df, cohort_df, feature,
                                       art["numeric_bin_edges"][feature])
        rows.append({"feature": feature, "feature_type": "numeric", **result})

    for feature in art["discrete_monitoring_cols"]:
        result = discrete_feature_drift(reference_df, cohort_df, feature)
        rows.append({"feature": feature, "feature_type": "discrete", **result})

    drift = pd.DataFrame(rows)
    drift["drift_level"] = drift["psi"].apply(classify_psi)
    drift["p_value_adj"] = benjamini_hochberg_adjustment(drift["p_value"].values)
    drift["supported"] = (drift["psi"] >= 0.10) & (drift["p_value_adj"] < 0.05)
    drift["supported_high"] = (drift["drift_level"] == "high") & drift["supported"]

    supported_count = int(drift["supported"].sum())
    supported_high_count = int(drift["supported_high"].sum())

    return drift, {"max_feature_psi": float(drift["psi"].max()),
                   "mean_feature_psi": float(drift["psi"].mean()),
                   "supported_feature_drift_count": supported_count,
                   "supported_high_drift_count": supported_high_count,
                   "feature_drift_trigger": bool(supported_high_count >= 1 or
                                                 supported_count >= 5)}


def calculate_jsd_from_counts(reference_counts, current_counts, epsilon=1e-12):
    reference = np.asarray(reference_counts, dtype=float) + epsilon
    current = np.asarray(current_counts, dtype=float) + epsilon

    reference = reference / reference.sum()
    current = current / current.sum()
    midpoint = 0.5 * (reference + current)

    kl_reference = np.sum(reference * np.log(reference / midpoint))
    kl_current = np.sum(current * np.log(current / midpoint))

    return 0.5 * (kl_reference + kl_current)


def data_quality_for_month(reference_df, cohort_df, art):
    """Unseen categorical levels and mean missingness, against the 2013 reference."""
    unseen_count = 0
    unseen_detail = []

    for feature in art["monitoring_categorical_cols"]:
        reference_levels = set(prepare_discrete_values(reference_df[feature]).unique())
        current_levels = set(prepare_discrete_values(cohort_df[feature]).unique())
        unseen = current_levels - reference_levels

        if unseen:
            unseen_count += len(unseen)
            unseen_detail.append(f"{feature}: {sorted(unseen)}")

    missing_rate = float(cohort_df[art["monitoring_feature_cols"]].isna().mean().mean())

    return {"unseen_category_count": int(unseen_count),
            "unseen_category_detail": "; ".join(unseen_detail),
            "mean_feature_missing_rate": missing_rate,
            "data_quality_flag": bool(unseen_count > 0 or
                                      missing_rate > art["data_quality_missing_rate_threshold"])}


def immediate_signals(reference_df, cohort_df, deployed_scores, champion_reference, art):
    """Everything available at monitor_month, for the champion that scored the cohort."""
    _, feature_summary = feature_drift_for_month(reference_df, cohort_df, art)

    counts, _ = np.histogram(deployed_scores, bins=champion_reference["score_edges"])
    prediction_jsd = float(calculate_jsd_from_counts(
        champion_reference["reference_counts"], counts))

    quality = data_quality_for_month(reference_df, cohort_df, art)

    signals = {**feature_summary,
               "prediction_jsd": prediction_jsd,
               "prediction_drift_trigger": bool(prediction_jsd >=
                                                art["prediction_jsd_moderate_band"]),
               **quality}

    signals["warning"] = bool(signals["feature_drift_trigger"] or
                              signals["prediction_drift_trigger"] or
                              signals["data_quality_flag"])
    return signals
