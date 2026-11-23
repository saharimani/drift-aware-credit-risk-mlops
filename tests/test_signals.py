"""Monitoring statistics: PSI, JSD, Benjamini-Hochberg and the feature-drift trigger."""

import numpy as np
import pandas as pd
import pytest

import signals as sig


# ------------------------------------------------------------------- PSI

def test_psi_is_zero_for_identical_distributions():
    counts = [100, 200, 300, 400]
    assert sig.calculate_psi_from_counts(counts, counts) == pytest.approx(0.0)


def test_psi_depends_on_proportions_not_sample_size():
    reference = [100, 200, 300, 400]
    current = [10, 20, 30, 40]  # same shape, ten times smaller
    assert sig.calculate_psi_from_counts(reference, current) == pytest.approx(0.0)


def test_psi_matches_hand_calculation():
    # reference 50/50, current 80/20
    expected = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert sig.calculate_psi_from_counts([50, 50], [80, 20]) == pytest.approx(expected)


def test_psi_survives_an_empty_bin():
    # A bin with zero count must not produce log(0) or inf
    value = sig.calculate_psi_from_counts([50, 50, 0], [40, 40, 20])
    assert np.isfinite(value) and value > 0


def test_psi_is_nan_when_a_side_is_empty():
    assert np.isnan(sig.calculate_psi_from_counts([0, 0], [1, 2]))


@pytest.mark.parametrize("psi, level", [(0.05, "low"), (0.10, "moderate"),
                                        (0.249, "moderate"), (0.25, "high"),
                                        (np.nan, "missing")])
def test_psi_bands(psi, level):
    assert sig.classify_psi(psi) == level


# ------------------------------------------------------------------- JSD

def test_jsd_is_zero_for_identical_distributions():
    assert sig.calculate_jsd_from_counts([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0, abs=1e-9)


def test_jsd_is_symmetric_and_bounded_by_log2():
    a, b = [90, 10, 0], [0, 10, 90]
    forward = sig.calculate_jsd_from_counts(a, b)
    backward = sig.calculate_jsd_from_counts(b, a)
    assert forward == pytest.approx(backward)
    assert 0 < forward <= np.log(2)


# ---------------------------------------------------- multiple testing (BH)

def test_bh_matches_statsmodels_reference_values():
    # Reference values from statsmodels.stats.multitest.multipletests(method="fdr_bh")
    p = [0.01, 0.04, 0.03, 0.20]
    expected = [0.04, 0.05333333, 0.05333333, 0.20]
    np.testing.assert_allclose(sig.benjamini_hochberg_adjustment(p), expected, rtol=1e-6)


def test_bh_keeps_nans_in_place_and_ignores_them_in_the_count():
    adjusted = sig.benjamini_hochberg_adjustment([0.01, np.nan, 0.02])
    assert np.isnan(adjusted[1])
    # two valid tests, so 0.01 * 2 / 1 and 0.02 * 2 / 2
    np.testing.assert_allclose(adjusted[[0, 2]], [0.02, 0.02])


def test_bh_adjusted_values_are_monotone_and_capped_at_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=50)
    adjusted = sig.benjamini_hochberg_adjustment(p)
    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)
    assert np.all(adjusted <= 1) and np.all(adjusted >= p - 1e-12)


# ---------------------------------------------------------- drift trigger

def _art(numeric, discrete):
    edges = {col: np.linspace(0, 1, 11) for col in numeric}
    return {"numeric_psi_cols": numeric, "numeric_bin_edges": edges,
            "discrete_monitoring_cols": discrete}


def test_no_trigger_when_the_cohort_looks_like_the_reference():
    rng = np.random.default_rng(1)
    cols = [f"x{i}" for i in range(6)]
    reference = pd.DataFrame(rng.uniform(size=(5000, 6)), columns=cols)
    cohort = pd.DataFrame(rng.uniform(size=(5000, 6)), columns=cols)

    _, summary = sig.feature_drift_for_month(reference, cohort, _art(cols, []))
    assert summary["feature_drift_trigger"] is False


def test_one_strongly_shifted_feature_triggers():
    rng = np.random.default_rng(2)
    cols = ["stable", "shifted"]
    reference = pd.DataFrame(rng.uniform(size=(5000, 2)), columns=cols)
    cohort = reference.copy()
    cohort["shifted"] = rng.uniform(0.6, 1.0, size=5000)  # mass moves to the top bins

    drift, summary = sig.feature_drift_for_month(reference, cohort, _art(cols, []))
    shifted = drift.set_index("feature").loc["shifted"]
    assert shifted["drift_level"] == "high" and bool(shifted["supported"])
    assert summary["supported_high_drift_count"] == 1
    assert summary["feature_drift_trigger"] is True


def test_discrete_drift_treats_missing_as_its_own_level():
    reference = pd.DataFrame({"grade": ["A"] * 50 + ["B"] * 50})
    cohort = pd.DataFrame({"grade": ["A"] * 50 + [None] * 50})
    result = sig.discrete_feature_drift(reference, cohort, "grade")
    assert result["psi"] > 0.25
