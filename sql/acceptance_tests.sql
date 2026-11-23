-- Acceptance tests. Each returns rows only when the test FAILS, except where noted.
--
-- Substitute PROJECT_ID and DATASET_ID before running, e.g.
--   sed -e "s/PROJECT_ID/${PROJECT}/g" -e "s/DATASET_ID/${DATASET}/g" \
--       sql/acceptance_tests.sql | bq query --use_legacy_sql=false --project_id="${PROJECT}"
--
-- Run after the full 41-month replay has completed for that dataset.
-- Every query is scoped to one dataset, so test and cloud results stay separate.

-- Test 0a (expects one row: months_scored = 36): every 2014-2016 cohort processed.
SELECT COUNT(DISTINCT monitor_month) AS months_scored
FROM `PROJECT_ID.DATASET_ID.monitoring_metrics`;

-- Test 0b (expects one row: requests = 33 of 36): the Policy A shadow baseline
-- reproduces the Notebook 05 request count.
SELECT COUNTIF(retraining_request) AS requests, COUNT(*) AS months
FROM `PROJECT_ID.DATASET_ID.policy_a_requests`;

-- Test 0c (expects 5 rows, one per trigger: 2014-11 R2, 2015-04 R3, 2015-11 R2,
-- 2016-05 R2, 2017-01 R3): the Policy C trigger history.
SELECT decision_month, fired_rule, intended_action, final_action,
       candidate_promoted, in_stream_decision
FROM `PROJECT_ID.DATASET_ID.policy_decisions`
WHERE candidate_trigger
ORDER BY decision_month;

-- Test 0d: no tail decision may be counted as an in-stream deployment.
-- FAILS if any 2017 decision claims in-stream status.
SELECT decision_month, in_stream_decision
FROM `PROJECT_ID.DATASET_ID.policy_decisions`
WHERE decision_month >= '2017-01' AND in_stream_decision = TRUE;

-- Test 0e (expects one row: challenger_at_2017-01 with zero cohorts): the tail
-- promotion has no deployment exposure.
SELECT p.champion_id, COUNT(s.monitor_month) AS cohorts_scored
FROM `PROJECT_ID.DATASET_ID.policy_promotions` p
LEFT JOIN `PROJECT_ID.DATASET_ID.prototype_scores` s
  ON s.scoring_champion_id = p.champion_id
WHERE p.decision_month != 'start'
  AND p.decision_month >= '2017-01'
GROUP BY p.champion_id;

-- Test 2: the delayed-label query never uses a label that had not arrived.
-- Every evidence row must draw on a cohort whose labels matured on or before the
-- decision month.
SELECT decision_month, monitor_month, label_available_month
FROM `PROJECT_ID.DATASET_ID.delayed_evidence`
WHERE label_available_month > decision_month;

-- Test 3: a cohort is scored only once, by exactly one champion.
SELECT monitor_month, COUNT(DISTINCT scoring_champion_id) AS champions,
       COUNT(DISTINCT run_id) AS scoring_runs
FROM `PROJECT_ID.DATASET_ID.prototype_scores`
GROUP BY monitor_month
HAVING champions > 1;

-- Test 4: the recorded scoring champion is the one the decision attributed the
-- evidence to. Any mismatch means evidence was measured against the wrong reference.
SELECT e.monitor_month, e.scored_by, s.scoring_champion_id
FROM `PROJECT_ID.DATASET_ID.delayed_evidence` e
JOIN (SELECT DISTINCT monitor_month, scoring_champion_id
      FROM `PROJECT_ID.DATASET_ID.prototype_scores`) s
  ON e.monitor_month = s.monitor_month
WHERE e.scored_by != s.scoring_champion_id;

-- Test 5: evidence from a superseded champion never triggers a candidate.
SELECT decision_month, monitor_month, scored_by, champion_id, fired_rule
FROM `PROJECT_ID.DATASET_ID.policy_decisions`
WHERE evidence_relevant = FALSE AND candidate_trigger = TRUE;

-- Test 5b (expects rows, and is the positive evidence for the same rule):
-- superseded cohorts are recorded but explicitly excluded.
SELECT decision_month, monitor_month, scored_by, champion_id, policy_reason
FROM `PROJECT_ID.DATASET_ID.policy_decisions`
WHERE evidence_relevant = FALSE
ORDER BY decision_month;

-- Test 6: candidate rejection does not change the active champion.
-- The champion recorded on a rejected decision must equal the champion recorded
-- on the next decision.
WITH ordered AS (
  SELECT decision_month, champion_id, candidate_promoted,
         LEAD(champion_id) OVER (ORDER BY decision_month) AS next_champion
  FROM `PROJECT_ID.DATASET_ID.policy_decisions`
)
SELECT * FROM ordered
WHERE candidate_promoted = FALSE
  AND next_champion IS NOT NULL
  AND next_champion != champion_id;

-- Test 7: a promotion changes the champion for the following cohort only.
-- The cohort scored in the promotion month keeps the previous champion.
SELECT p.decision_month, p.champion_id,
       MIN(s.monitor_month) AS first_cohort_scored
FROM `PROJECT_ID.DATASET_ID.policy_promotions` p
JOIN `PROJECT_ID.DATASET_ID.prototype_scores` s
  ON s.scoring_champion_id = p.champion_id
WHERE p.decision_month != 'start'
GROUP BY p.decision_month, p.champion_id
HAVING first_cohort_scored <= p.decision_month;

-- Test 8: duplicate execution creates no duplicate decisions.
SELECT policy_name, policy_version, decision_month, COUNT(*) AS n
FROM `PROJECT_ID.DATASET_ID.policy_decisions`
GROUP BY 1, 2, 3
HAVING n > 1;

-- Test 8b (expects at least one row): the skipped duplicate run was recorded.
SELECT run_id, decision_month, outcome, job_duration_seconds
FROM `PROJECT_ID.DATASET_ID.prototype_runs`
WHERE outcome = 'duplicate_run_skipped';

-- Test 9: policy state survives between executions and is a single row.
SELECT COUNT(*) AS state_rows
FROM `PROJECT_ID.DATASET_ID.policy_state`;

-- Runtime evidence for the cost section.
SELECT outcome, candidate_type, candidate_promoted,
       COUNT(*) AS runs,
       ROUND(AVG(job_duration_seconds), 1) AS mean_job_seconds,
       ROUND(AVG(monitoring_duration_seconds), 1) AS mean_monitoring_seconds,
       ROUND(AVG(candidate_fit_duration_seconds), 1) AS mean_fit_seconds
FROM `PROJECT_ID.DATASET_ID.prototype_runs`
GROUP BY 1, 2, 3
ORDER BY 1, 2;
