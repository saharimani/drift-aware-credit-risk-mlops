# Cloud Storage artefacts, BigQuery writes and structured logging.
# Everything loaded here is an artefact already produced by Notebooks 03, 04 or 05.

import io
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

from google.cloud import bigquery, storage


def load_config(path=None):
    path = path or os.environ.get("PROTOTYPE_CONFIG", "config/prototype_config.json")

    with open(path) as f:
        cfg = json.load(f)

    # Environment wins, so one image can run against different projects
    for key in ["project_id", "bucket", "dataset", "policy_name", "policy_version"]:
        cfg[key] = os.environ.get(key.upper(), cfg[key])

    os.makedirs(cfg["cache_dir"], exist_ok=True)
    return cfg


def log(event, severity="INFO", **fields):
    """One line of JSON per event, so Cloud Logging parses it into jsonPayload."""
    record = {"severity": severity, "event": event, **fields}
    print(json.dumps(record, default=str), flush=True)


# ---------------------------------------------------------------- Cloud Storage

_gcs_client = None


def gcs():
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client


def fetch(cfg, blob_name):
    """Download once per container, then reuse from the local cache."""
    local_path = os.path.join(cfg["cache_dir"], blob_name.replace("/", "__"))

    if not os.path.exists(local_path):
        gcs().bucket(cfg["bucket"]).blob(blob_name).download_to_filename(local_path)
        log("artefact_downloaded", blob=blob_name, local_path=local_path)

    return local_path


def upload(cfg, local_path, blob_name):
    gcs().bucket(cfg["bucket"]).blob(blob_name).upload_from_filename(local_path)
    uri = f"gs://{cfg['bucket']}/{blob_name}"
    log("artefact_uploaded", uri=uri)
    return uri


def read_uri(cfg, uri):
    """Load a joblib artefact from a gs:// URI held in policy_state."""
    prefix = f"gs://{cfg['bucket']}/"
    if not uri.startswith(prefix):
        raise ValueError(f"URI is outside the prototype bucket: {uri}")
    return joblib.load(fetch(cfg, uri[len(prefix):]))


def col_list(cfg, key):
    return pd.read_csv(fetch(cfg, cfg["gcs"][key]))["feature"].tolist()


def load_json(cfg, key):
    with open(fetch(cfg, cfg["gcs"][key])) as f:
        return json.load(f)


def load_artefacts(cfg):
    """Feature lists, reference bins and the thresholds fixed in Notebooks 04 and 05."""
    monitoring = load_json(cfg, "monitoring_metadata")
    policy = load_json(cfg, "policy_metadata")

    art = {
        "champion_feature_cols": col_list(cfg, "champion_feature_cols"),
        "monitoring_feature_cols": col_list(cfg, "monitoring_feature_cols"),
        "monitoring_numeric_cols": col_list(cfg, "monitoring_numeric_cols"),
        "monitoring_categorical_cols": col_list(cfg, "monitoring_categorical_cols"),
        "numeric_psi_cols": col_list(cfg, "numeric_psi_cols"),
        "discrete_monitoring_cols": col_list(cfg, "discrete_monitoring_cols"),
        "numeric_bin_edges": joblib.load(fetch(cfg, cfg["gcs"]["numeric_bin_edges"])),
        "score_edges": joblib.load(fetch(cfg, cfg["gcs"]["score_bin_edges"])),
        "monitoring_metadata": monitoring,
        "policy_metadata": policy,
        "gate": policy["deployment_gates"][cfg["gate_name"]],
        "fork": policy["action_fork_thresholds"],
        "label_delay_months": int(policy["label_delay_months"]),
        "validation_window_months": int(policy["validation_window_months"]),
        "deployment_cooldown_months": int(policy["deployment_cooldown_months"]),
        "candidate_evaluation_holdoff_months": int(policy["candidate_evaluation_holdoff_months"]),
        "persistence_cohorts": int(policy["persistence_cohorts"]),
    }

    # Thresholds are read, never restated, so the cloud cannot drift from the notebooks
    for key in ["prediction_jsd_moderate_band", "concept_moderate_threshold",
                "concept_high_threshold", "calibration_gap_moderate_threshold",
                "calibration_gap_high_threshold", "brier_moderate_threshold",
                "brier_high_threshold", "auc_drop_moderate_threshold",
                "auc_drop_high_threshold", "data_quality_missing_rate_threshold"]:
        art[key] = float(monitoring[key])

    return art


_frames = {}


def load_frames(cfg):
    """The two processed parquet files, exactly as Notebooks 04 and 05 read them.

    Read whole rather than split per month: the categorical dtypes and their level
    sets are defined by the file schema, and splitting risks changing them.
    """
    if not _frames:
        historical = pd.read_parquet(fetch(cfg, cfg["gcs"]["historical_parquet"]))
        simulation = pd.read_parquet(fetch(cfg, cfg["gcs"]["simulation_parquet"]))
        simulation["monitor_month"] = (pd.to_datetime(simulation["issue_d"])
                                       .dt.to_period("M").astype(str))

        _frames["historical"] = historical
        _frames["simulation"] = simulation
        _frames["months"] = sorted(simulation["monitor_month"].unique())

        log("cohort_data_loaded", historical_rows=len(historical),
            simulation_rows=len(simulation), months=len(_frames["months"]))

    return _frames


def load_cohort(cfg, month):
    frames = load_frames(cfg)
    return frames["simulation"][frames["simulation"]["monitor_month"] == month]


# ------------------------------------------------------------------- BigQuery

_bq_client = None


def bq(cfg):
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=cfg["project_id"])
    return _bq_client


def table_id(cfg, table):
    return f"{cfg['project_id']}.{cfg['dataset']}.{table}"


def query(cfg, sql, **params):
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter(name, "STRING", value)
        for name, value in params.items()])

    return bq(cfg).query(sql, job_config=job_config).result().to_dataframe()


def merge_rows(cfg, table, rows, keys):
    """Idempotently MERGE rows using the destination table's declared schema.

    Reusing the target schema prevents all-null dataframe columns from being
    assigned an incorrect staging type by pandas or PyArrow.
    """
    if isinstance(rows, list):
        rows = pd.DataFrame(rows)

    if rows.empty:
        return 0

    target = table_id(cfg, table)
    staging = f"{target}__staging"
    client = bq(cfg)

    target_table = client.get_table(target)
    schema_by_name = {field.name: field for field in target_table.schema}

    unknown_columns = [
        column for column in rows.columns
        if column not in schema_by_name
    ]
    if unknown_columns:
        raise ValueError(
            f"Columns are absent from {target}: {unknown_columns}"
        )

    missing_keys = [key for key in keys if key not in rows.columns]
    if missing_keys:
        raise ValueError(f"MERGE keys are absent from dataframe: {missing_keys}")

    staging_schema = [
        schema_by_name[column]
        for column in rows.columns
    ]

    load_config = bigquery.LoadJobConfig(
        schema=staging_schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
    )

    def qid(name):
        name = str(name)
        if "`" in name:
            raise ValueError(f"Invalid BigQuery identifier: {name}")
        return f"`{name}`"

    client.delete_table(staging, not_found_ok=True)

    try:
        client.load_table_from_dataframe(
            rows,
            staging,
            job_config=load_config,
        ).result()

        on_clause = " AND ".join(
            f"T.{qid(key)} = S.{qid(key)}"
            for key in keys
        )

        columns = list(rows.columns)
        updates = ", ".join(
            f"{qid(column)} = S.{qid(column)}"
            for column in columns
            if column not in keys
        )
        insert_columns = ", ".join(qid(column) for column in columns)
        insert_values = ", ".join(
            f"S.{qid(column)}"
            for column in columns
        )

        matched_clause = (
            f"WHEN MATCHED THEN UPDATE SET {updates}"
            if updates else ""
        )

        sql = f"""
        MERGE `{target}` T
        USING `{staging}` S
        ON {on_clause}
        {matched_clause}
        WHEN NOT MATCHED THEN INSERT ({insert_columns})
          VALUES ({insert_values})
        """

        client.query(sql).result()

    finally:
        client.delete_table(staging, not_found_ok=True)

    log("bigquery_merge", table=table, rows=len(rows), keys=keys)
    return len(rows)

