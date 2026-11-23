# Candidate construction.
# Copied from Notebook 05 cells 54-55. The only change is that historical_df,
# the champion feature list and n_jobs are arguments rather than globals.

import pandas as pd
from xgboost import XGBClassifier

from evidence import fit_recalibration_map, score_raw


def build_recalibration_candidate(champion_state, training_df, decision_month,
                                  champion_feature_cols):
    raw_training_scores = score_raw(champion_state, training_df, champion_feature_cols)
    replacement_map = fit_recalibration_map(raw_training_scores,
                                            training_df["target_bad_loan"].values)

    return {"champion_id": f"recalibrated_at_{decision_month}",
            "underlying_model": champion_state["underlying_model"],
            "active_recalibration_map": replacement_map,
            "deployment_month": decision_month,
            "action_type": "recalibration",
            "reference": None}


def build_challenger_candidate(historical_df, training_df, decision_month,
                               champion_feature_cols, n_jobs=4):
    challenger_train_df = pd.concat([historical_df, training_df[historical_df.columns]],
                                    ignore_index=True)

    challenger_model = XGBClassifier(objective="binary:logistic",
                                     eval_metric="logloss",
                                     tree_method="hist",
                                     enable_categorical=True,
                                     n_estimators=500,
                                     learning_rate=0.05,
                                     max_depth=4,
                                     min_child_weight=20,
                                     subsample=0.8,
                                     colsample_bytree=0.8,
                                     reg_lambda=2.0,
                                     random_state=42,
                                     n_jobs=n_jobs)

    challenger_model.fit(challenger_train_df[champion_feature_cols],
                         challenger_train_df["target_bad_loan"],
                         verbose=False)

    return {"champion_id": f"challenger_at_{decision_month}",
            "underlying_model": challenger_model,
            "active_recalibration_map": None,
            "deployment_month": decision_month,
            "action_type": "challenger",
            "reference": None}
