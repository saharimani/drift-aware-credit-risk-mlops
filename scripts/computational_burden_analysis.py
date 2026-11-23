"""
computational_burden_analysis.py

Reusable calculation helpers for 06_computational_burden_audit.ipynb.

"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Expected input inventory


OFFLINE_FILES = {"policy_A_requests.csv": "Policy A monthly feature-drift request record (request stage only).",
                 "policy_B_candidate_attempts.csv": "Policy B candidate fits with training-window row counts.",
                 "policy_C_candidate_attempts.csv": "Policy C candidate fits (challengers + recalibration).",
                 "policy_D_candidate_attempts.csv": "Policy D scheduled candidate fits.",
                 "policy_B_decisions.csv": "Policy B monthly decision log (triggers, suppressions).",
                 "policy_C_decisions.csv": "Policy C monthly decision log (rules, suppressions).",
                 "policy_D_decisions.csv": "Policy D scheduled decision log.",
                 "policy_B_promotions.csv": "Policy B promotions (deployment stage only).",
                 "policy_C_promotions.csv": "Policy C promotions (deployment stage only).",
                 "policy_D_promotions.csv": "Policy D promotions (deployment stage only).",
                 "final_policy_metadata.json": "Study configuration and final headline policy results.",
                 "monthly_counts.csv": "Monthly scoring-cohort row counts, 2014-01 to 2016-12.",
                 "split_summary.csv": "Historical (2013) and simulation (2014-2016) split row counts.",}

GCP_FILES = {"prototype_runs.json": "Per-execution Cloud Run audit record (durations, outcome).",
             "candidate_attempts.json": "Cloud candidate fits with training rows and gate outcome.",
             "policy_decisions.json": "Cloud Policy C monthly decision records.",
             "policy_promotions.json": "Cloud promotion export.",
             "policy_a_requests.json": "Cloud Policy A shadow request records.",
             "monitoring_metrics.json": "Cloud monitoring metrics per scoring month.",
             "monitor-and-decide.yaml": "Cloud Run Job definition (CPU / memory allocation).",
             "executions.json": "Cloud Run platform execution list (independent of the app log).",}

# Required input files
REQUIRED_OFFLINE = set(OFFLINE_FILES)
REQUIRED_GCP = set(GCP_FILES)


# Convert a YYYY-MM string to a numeric month index
def month_to_index(month: str) -> int:
    year, mon = month.split("-")
    return int(year) * 12 + int(mon) - 1


# Convert a numeric month index back to YYYY-MM
def index_to_month(idx: int) -> str:
    year, mon = divmod(idx, 12)
    return f"{year:04d}-{mon + 1:02d}"


# Shift a month forwards or backwards
def shift_month(month: str, delta: int) -> str:
    return index_to_month(month_to_index(month) + delta)


# Return all months between two dates
def months_between(start: str, end: str) -> list[str]:
    a, b = month_to_index(start), month_to_index(end)
    return [index_to_month(i) for i in range(a, b + 1)] if b >= a else []


# Count rows in CSV files and records in JSON files
def _record_count(path: Path):
    try:
        if path.suffix == ".csv":
            return len(pd.read_csv(path))
        if path.suffix == ".json":
            with path.open() as fh:
                obj = json.load(fh)
            return len(obj) if isinstance(obj, list) else "1 (object)"
    except Exception as exc:
        return f"unreadable: {type(exc).__name__}"
    return "-"


# Build the input-file summary
def build_input_inventory(inputs_dir: Path) -> pd.DataFrame:
    rows = []
    for category, folder, spec in (("offline", inputs_dir / "offline", OFFLINE_FILES), ("gcp", inputs_dir / "gcp", GCP_FILES)):
        for filename, purpose in spec.items():
            path = folder / filename
            exists = path.exists()
            rows.append({"category": category, "filename": filename, "exists": exists, "size_bytes": path.stat().st_size if exists else 0, "records": _record_count(path) if exists else "-", "purpose": purpose})
    return pd.DataFrame(rows).sort_values(["category", "filename"]).reset_index(drop=True)


# Check that all required files are present
def check_inventory(inventory: pd.DataFrame) -> None:
    missing = inventory.loc[~inventory["exists"], ["category", "filename"]]
    if len(missing):
        listing = "\n".join(f"  - {r.category}/{r.filename}" for r in missing.itertuples())
        raise FileNotFoundError(f"Required input files are missing:\n{listing}")


# Load a JSON file
def load_json(path: Path):
    with Path(path).open() as fh:
        return json.load(fh)


# Load a GCP JSON export as a dataframe
def load_gcp_table(path: Path) -> pd.DataFrame:
    return pd.DataFrame(load_json(path)).astype("string")



# Study configuration
@dataclass(frozen=True)
class StudyConfig:
    label_delay_months: int
    validation_window_months: int
    deployment_cooldown_months: int
    candidate_evaluation_holdoff_months: int

    @classmethod
    def from_metadata(cls, metadata: dict) -> "StudyConfig":
        return cls(label_delay_months=int(metadata["label_delay_months"]),
                   validation_window_months=int(metadata["validation_window_months"]),
                   deployment_cooldown_months=int(metadata["deployment_cooldown_months"]),
                   candidate_evaluation_holdoff_months=int(metadata["candidate_evaluation_holdoff_months"]))


# Summarise candidate-fitting workload for one policy
def summarise_candidate_attempts(attempts: pd.DataFrame, policy_label: str) -> dict:
    challengers = attempts.loc[attempts["candidate_type"] == "challenger"]
    recalibrations = attempts.loc[attempts["candidate_type"] == "recalibration"]
    rows = attempts["total_training_rows"]

    return {"policy": policy_label, "candidate_fits": int(len(attempts)), "challenger_fits": int(len(challengers)),
            "recalibration_fits": int(len(recalibrations)), "candidate_training_row_exposure": int(rows.sum()),
            "challenger_training_row_exposure": int(challengers["total_training_rows"].sum()),
            "recalibration_training_row_exposure": int(recalibrations["total_training_rows"].sum()), "mean_rows_per_fit":
            float(rows.mean()), "median_rows_per_fit": float(rows.median()), "min_rows_per_fit": int(rows.min()),
            "max_rows_per_fit": int(rows.max())}


# Reconstruct the training and validation window for a decision month
def training_design_for_decision_month(decision_month: str, config: StudyConfig) -> dict:
    validation_end = shift_month(decision_month, -config.label_delay_months)
    validation_start = shift_month(validation_end, -(config.validation_window_months - 1))
    training_end = shift_month(validation_start, -1)

    return {"validation_start": validation_start, "validation_end": validation_end, "training_end": training_end}


# Check the reconstructed windows against the observed candidate fits
def verify_training_design(attempts: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    rows = []

    for rec in attempts.itertuples():
        design = training_design_for_decision_month(rec.decision_month, config)
        rows.append({"policy": rec.policy, "decision_month": rec.decision_month, "candidate_type": rec.candidate_type, "observed_training_end": rec.simulation_training_end_month, "reconstructed_training_end": design["training_end"], "observed_validation_start": rec.validation_start_month, "reconstructed_validation_start": design["validation_start"], "observed_validation_end": rec.validation_end_month, "reconstructed_validation_end": design["validation_end"]})

    out = pd.DataFrame(rows)
    out["design_matches"] = (out["observed_training_end"] == out["reconstructed_training_end"]) & (out["observed_validation_start"] == out["reconstructed_validation_start"]) & (out["observed_validation_end"] == out["reconstructed_validation_end"])

    return out


# Reconstruct the Policy A* counterfactual workload
def reconstruct_policy_a_counterfactual(requests: pd.DataFrame, monthly_counts: pd.DataFrame, historical_rows: int, config: StudyConfig) -> pd.DataFrame:
    counts = monthly_counts.set_index("monitor_month")["rows"].astype(int)
    simulation_start = counts.index.min()
    simulation_end = counts.index.max()
    rows = []

    for rec in requests.itertuples():
        month = rec.monitor_month
        design = training_design_for_decision_month(month, config)
        training_end = design["training_end"]
        validation_end = design["validation_end"]

        window = months_between(simulation_start, training_end)
        has_training_month = len(window) > 0
        has_validation = month_to_index(validation_end) <= month_to_index(simulation_end)
        feasible = has_training_month and has_validation

        if not has_training_month:
            reason = f"no matured training month: training window would end {training_end}, before the simulation starts ({simulation_start})"
        elif not has_validation:
            reason = f"validation window ends {validation_end}, beyond the scoring period ({simulation_end})"
        else:
            reason = ""

        simulation_rows = int(counts.loc[window].sum()) if feasible else 0
        rows.append({"request_month": month, "feasible_fit": feasible, "infeasible_reason": reason, "training_start": simulation_start if feasible else "", "training_end": training_end if feasible else "", "validation_start": design["validation_start"] if feasible else "", "validation_end": validation_end if feasible else "", "historical_training_rows": historical_rows if feasible else 0, "simulation_training_rows": simulation_rows, "total_training_rows": historical_rows + simulation_rows if feasible else 0})

    return pd.DataFrame(rows).sort_values("request_month").reset_index(drop=True)


# Duplicate Cloud Run outcome
DUPLICATE_OUTCOME = "duplicate_run_skipped"


# Numeric columns in the Cloud Run records
RUN_NUMERIC_COLUMNS = ["job_duration_seconds", "monitoring_duration_seconds",
                       "candidate_fit_duration_seconds","rows_scored"]


# Load and prepare the Cloud Run replay records
def prepare_prototype_runs(path: Path) -> pd.DataFrame:
    runs = load_gcp_table(path)

    for column in RUN_NUMERIC_COLUMNS:
        runs[column] = pd.to_numeric(runs[column])

    runs["candidate_fit_started"] = runs["candidate_fit_started"] == "true"
    runs["candidate_promoted"] = runs["candidate_promoted"] == "true"

    return runs.sort_values(["started_at", "decision_month"]).reset_index(drop=True)


# Separate the main replay from duplicate-run records
def split_main_and_duplicate_runs(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    is_duplicate = runs["outcome"] == DUPLICATE_OUTCOME
    return runs.loc[~is_duplicate].copy(), runs.loc[is_duplicate].copy()


# Give each replay execution a simple category
def classify_execution(row) -> str:
    if row["candidate_fit_started"]:
        return "candidate-fit month"
    if row["rows_scored"] == 0:
        return "maturity-only tail month"
    if row["outcome"] == "no_decision":
        return "early month, no decision yet"

    return "monitoring and decision, no candidate fit"


# Summarise measured Cloud Run runtime
def summarise_cloud_runtime(main_runs: pd.DataFrame) -> dict:
    duration = main_runs["job_duration_seconds"]

    return {"main_executions": int(len(main_runs)), "scoring_executions": int((main_runs["rows_scored"] > 0).sum()),
            "tail_executions": int((main_runs["rows_scored"] == 0).sum()), "total_wall_clock_seconds":
            float(duration.sum()), "mean_wall_clock_seconds": float(duration.mean()), "median_wall_clock_seconds":
            float(duration.median()), "min_wall_clock_seconds": float(duration.min()), "max_wall_clock_seconds":
            float(duration.max()), "total_monitoring_seconds": float(main_runs["monitoring_duration_seconds"].sum()),
            "total_candidate_fit_seconds": float(main_runs["candidate_fit_duration_seconds"].sum()), "total_rows_scored":
            int(main_runs["rows_scored"].sum())}


# Pattern used to read the Cloud Run memory value
_MEMORY_PATTERN = re.compile(r"^\s*(?P<value>[0-9.]+)\s*(?P<unit>[KMGT]i?)?[Bb]?\s*$")


# Convert Cloud Run memory units to GiB
_MEMORY_UNITS_IN_GIB = {None: 1 / (1024 ** 3), "K": 1000 / (1024 ** 3), "Ki": 1 / (1024 ** 2), "M": 1e6 / (1024 ** 3),
                        "Mi": 1 / 1024, "G": 1e9 / (1024 ** 3), "Gi": 1.0, "T": 1e12 / (1024 ** 3), "Ti": 1024.0}


# Convert the configured memory value to GiB
def parse_memory_to_gib(text: str) -> float:
    match = _MEMORY_PATTERN.match(str(text))

    if not match:
        raise ValueError(f"Unrecognised Cloud Run memory value: {text!r}")

    return float(match.group("value")) * _MEMORY_UNITS_IN_GIB[match.group("unit")]


# Read the Cloud Run CPU and memory allocation
def read_cloud_run_allocation(yaml_path: Path) -> dict:
    import yaml

    with Path(yaml_path).open() as fh:
        spec = yaml.safe_load(fh)

    container = spec["spec"]["template"]["spec"]["template"]["spec"]["containers"][0]
    limits = container["resources"]["limits"]

    return {"job_name": spec["metadata"]["name"], "region": spec["metadata"]["labels"]["cloud.googleapis.com/location"],
            "cpu_allocated": float(limits["cpu"]), "memory_allocated_raw": str(limits["memory"]), "memory_allocated_gib":
            parse_memory_to_gib(limits["memory"]), "task_count": int(spec["spec"]["template"]["spec"]["taskCount"]),
            "max_retries": int(spec["spec"]["template"]["spec"]["template"]["spec"]["maxRetries"])}


# Calculate allocated CPU and memory time
def allocation_time_proxy(seconds: float, cpu: float, memory_gib: float) -> dict:
    return {"wall_clock_seconds": float(seconds), "vcpu_seconds": float(seconds * cpu),
            "vcpu_hours": float(seconds * cpu / 3600), "gib_seconds": float(seconds * memory_gib),
            "gib_hours": float(seconds * memory_gib / 3600)}


# Summarise the Cloud Run platform execution list
def summarise_cloud_run_executions(path: Path) -> pd.DataFrame:
    rows = []

    for execution in load_json(path):
        metadata = execution.get("metadata", {})
        status = execution.get("status", {})
        conditions = {c.get("type"): c.get("status") for c in status.get("conditions", [])}
        rows.append({"execution_name": metadata.get("name"), "created_at": metadata.get("creationTimestamp"), "start_time":
                     status.get("startTime"), "completion_time": status.get("completionTime"), "succeeded_count":
                     status.get("succeededCount") or 0, "failed_count": status.get("failedCount") or 0,
                     "completed_condition": conditions.get("Completed")})
        executions = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
        executions["succeeded"] = executions["completed_condition"] == "True"

    return executions


# Fit a simple linear runtime model
def fit_linear_model(x: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.linear_model import LinearRegression

    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    model = LinearRegression().fit(x, y)

    return {"specification": "seconds ~ intercept + slope * million_rows", "intercept": float(model.intercept_), "slope": float(model.coef_[0]), "r_squared": float(model.score(x, y)), "n_observations": int(len(y))}


# Fit a runtime model through the origin
def fit_through_origin_model(x: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.linear_model import LinearRegression

    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    model = LinearRegression(fit_intercept=False).fit(x, y)

    return {"specification": "seconds ~ slope * million_rows (through origin)", "intercept": 0.0, "slope": float(model.coef_[0]), "r_squared": float(model.score(x, y)), "n_observations": int(len(y))}


# Estimate challenger runtime from training rows
def predict_runtime(model: dict, training_rows) -> np.ndarray:
    million_rows = np.asarray(training_rows, dtype=float) / 1_000_000
    return model["intercept"] + model["slope"] * million_rows


# Calculate the percentage reduction from a baseline
def percentage_reduction(baseline: float, comparison: float) -> float:
    return (baseline - comparison) / baseline * 100
