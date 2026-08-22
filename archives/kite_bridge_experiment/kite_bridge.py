def get_ranked_signals(self, dfs_dict, long_threshold=None):
    from feature_engineer import FeatureEngineer
    import config
    import pandas as pd

    all_data = pd.concat(dfs_dict.values()).reset_index(drop=True)

    fe = FeatureEngineer(all_data)
    all_data = fe.generate_features()
    all_data = fe.rank_features()

    latest_batch = all_data.groupby('symbol').tail(1)

    feature_list = fe.get_feature_list()

    latest_batch["prob"] = self.model.predict_proba(
        latest_batch[feature_list]
    )[:, 1]

    latest_batch = latest_batch.sort_values("prob", ascending=False)
    latest_batch["rank"] = range(1, len(latest_batch)+1)

    signals = {}

    for _, row in latest_batch.iterrows():
        symbol = row['symbol']
        prob = row['prob']
        regime = row.get('regime', 1)

        if (
            prob >= config.LONG_THRESHOLD and
            row["rank"] <= config.MAX_POSITIONS and
            regime == 1
        ):
            action = "BUY"
        else:
            action = "HOLD"

        signals[symbol] = {
            "action": action,
            "prob": prob,
            "rank": row["rank"]
        }

    return signals