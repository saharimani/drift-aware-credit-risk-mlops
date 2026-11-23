# Full chronological Policy C replay: 2014-01 to 2016-12 scoring months, then
# 2017-01 to 2017-05 tail decision months. 41 executions in total.
#
# Every environment value supplied here is propagated to every execution, so the
# script can never read completed months from one dataset and write results to
# another, and can never run under a policy_version other than the one given.
#
#   Test replay (containerised code, still reads/writes GCS and BigQuery):
#     python scripts/replay_full_policy_c.py --local \
#       --config config/prototype_config_test.json \
#       --project PROJECT --bucket BUCKET \
#       --dataset drift_prototype_test --policy-version v1_primary_gate_delay5_test
#
#   Cloud replay:
#     python scripts/replay_full_policy_c.py \
#       --config config/prototype_config_cloud.json \
#       --project PROJECT --bucket BUCKET --region europe-west2 --job monitor-and-decide \
#       --dataset drift_prototype_cloud --policy-version v1_primary_gate_delay5_cloud
#
# --local means the Python process runs here (or inside the local container)
# rather than on Cloud Run. It is not offline: it still reads Cloud Storage and
# reads and writes BigQuery.
#
# Recovery: fully completed months are skipped. Recovery from a partially
# written month is outside the prototype scope; restart a failed validation
# replay against a clean test dataset.

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "src")

SCORING_MONTHS = [str(p) for p in pd.period_range("2014-01", "2016-12", freq="M")]
TAIL_MONTHS = [str(p) for p in pd.period_range("2017-01", "2017-05", freq="M")]
ALL_MONTHS = SCORING_MONTHS + TAIL_MONTHS

TERMINAL_OUTCOMES = ("completed", "no_decision")


def resolve_config(args):
    """One resolved environment, used for the completed-months query, for the
    local subprocess environment and for the Cloud Run execution overrides."""
    import storage

    cfg = storage.load_config(args.config)

    for key, value in [("project_id", args.project), ("bucket", args.bucket),
                       ("dataset", args.dataset), ("policy_name", args.policy),
                       ("policy_version", args.policy_version), ("region", args.region)]:
        if value:
            cfg[key] = value

    missing = [k for k in ["project_id", "bucket", "dataset", "policy_name", "policy_version"]
               if str(cfg.get(k, "")).startswith("REPLACE_WITH") or not cfg.get(k)]

    if missing:
        raise SystemExit(f"Unresolved configuration values: {missing}. "
                         "Fill them in the config file or pass them as arguments.")

    return cfg


def completed_months(cfg):
    """Months already carrying a terminal run record.

    A BigQuery failure is raised, never swallowed: a missing dataset, bad
    credentials or a permission error must stop the replay rather than be read
    as 'zero completed months', which would silently restart from 2014-01.
    """
    import storage

    sql = f"""
    SELECT DISTINCT decision_month
    FROM `{storage.table_id(cfg, 'prototype_runs')}`
    WHERE policy_name = @policy_name
      AND policy_version = @policy_version
      AND outcome IN ('completed', 'no_decision')
    """
    done = storage.query(cfg, sql,
                         policy_name=cfg["policy_name"],
                         policy_version=cfg["policy_version"])

    return set(done["decision_month"].tolist())


def environment_for(cfg):
    """Environment variables read by storage.load_config inside the job."""
    return {"PROJECT_ID": cfg["project_id"],
            "BUCKET": cfg["bucket"],
            "DATASET": cfg["dataset"],
            "POLICY_NAME": cfg["policy_name"],
            "POLICY_VERSION": cfg["policy_version"]}


def run_local(month, cfg, args):
    env = dict(os.environ)
    env.update(environment_for(cfg))

    if args.config:
        env["PROTOTYPE_CONFIG"] = args.config

    command = [sys.executable, "src/main.py", "--month", month,
               "--policy", cfg["policy_name"]]

    if args.config:
        command += ["--config", args.config]

    return subprocess.run(command, env=env).returncode


def run_cloud(month, cfg, args):
    overrides = ",".join(f"{k}={v}" for k, v in environment_for(cfg).items())

    command = ["gcloud", "run", "jobs", "execute", args.job,
               f"--region={cfg['region']}", f"--project={cfg['project_id']}",
               f"--args=--month={month},--policy={cfg['policy_name']}",
               f"--update-env-vars={overrides}",
               "--wait"]

    return subprocess.run(command).returncode


def main():
    parser = argparse.ArgumentParser(description="Replay Policy C month by month.")
    parser.add_argument("--config", default=None,
                        help="path to the config file for this environment")
    parser.add_argument("--project", default=None)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--policy-version", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--job", default="monitor-and-decide")
    parser.add_argument("--local", action="store_true",
                        help="run in this Python process instead of on Cloud Run; "
                             "still reads Cloud Storage and BigQuery")
    args = parser.parse_args()

    cfg = resolve_config(args)

    print("Replay environment:")
    print(json.dumps({"mode": "local" if args.local else "cloud-run",
                      "config_file": args.config or os.environ.get(
                          "PROTOTYPE_CONFIG", "config/prototype_config.json"),
                      **environment_for(cfg),
                      "region": cfg.get("region"),
                      "job": None if args.local else args.job}, indent=2))

    done = completed_months(cfg)
    print(f"\nAlready completed: {len(done)} of {len(ALL_MONTHS)} months")

    for month in ALL_MONTHS:
        phase = "scoring" if month in SCORING_MONTHS else "tail"

        if month in done:
            print(f"skip {month} ({phase}): already completed")
            continue

        print(f"\n=== {month} ({phase}) ===")
        started = time.time()
        code = run_local(month, cfg, args) if args.local else run_cloud(month, cfg, args)
        print(f"    {month} finished in {time.time() - started:.0f}s, exit {code}")

        if code != 0:
            print(f"\nFAILED at {month}.")
            print("Fully completed months are skipped on restart. If this month wrote "
                  "some rows before failing, that partial state is not repaired "
                  "automatically: restart the validation replay against a clean "
                  "test dataset rather than resuming over it.")
            sys.exit(1)

    print(f"\nReplay complete: {len(SCORING_MONTHS)} scoring months and "
          f"{len(TAIL_MONTHS)} tail months processed in "
          f"{cfg['dataset']} under {cfg['policy_version']}.")


if __name__ == "__main__":
    main()
