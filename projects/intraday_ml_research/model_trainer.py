import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import config


class ModelTrainer:
    def __init__(self, data, feature_cols, target_col='label'):
        self.df = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.model = None

    def train_test_split(self):
        unique_dates = sorted(self.df['date'].unique())
        split_idx = int(len(unique_dates) * config.TRAIN_RATIO)
        split_date = unique_dates[split_idx]

        train_df = self.df[self.df['date'] < split_date].copy()
        test_df = self.df[self.df['date'] >= split_date].copy()

        print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
        return train_df, test_df

    def train(self):
        params = config.XGB_PARAMS.copy()
        train_df, test_df = self.train_test_split()

        unique_dates = sorted(train_df['date'].unique())
        val_idx = int(len(unique_dates) * 0.85)
        calib_idx = int(len(unique_dates) * 0.90)

        val_date = unique_dates[val_idx]
        calib_date = unique_dates[calib_idx]

        tr = train_df[train_df['date'] < val_date]
        val = train_df[(train_df['date'] >= val_date) & (train_df['date'] < calib_date)]
        calib = train_df[train_df['date'] >= calib_date]

        X_train = tr[self.feature_cols]
        y_train = (tr[self.target_col] == 1).astype(int)

        X_val = val[self.feature_cols]
        y_val = (val[self.target_col] == 1).astype(int)

        X_calib = calib[self.feature_cols]
        y_calib = (calib[self.target_col] == 1).astype(int)

        X_test = test_df[self.feature_cols]
        y_test = (test_df[self.target_col] == 1).astype(int)

        base_model = xgb.XGBClassifier(**params)
        base_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        print(f"\nCalibrating model on {len(X_calib)} rows...")
        calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
        calibrated.fit(X_calib, y_calib)

        self.model = calibrated

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        print(classification_report(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))

        return test_df

    def save_model(self, path):
        joblib.dump(self.model, path)
        joblib.dump(self.feature_cols, 'feature_cols.joblib')

    def get_feature_importance(self):
        try:
            base_model = self.model.calibrated_classifiers_[0].estimator
            importance = base_model.feature_importances_
        except Exception:
            print("Feature importance not available")
            return

        feat_imp = pd.Series(importance, index=self.feature_cols).sort_values(ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x=feat_imp.values[:15], y=feat_imp.index[:15])
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Saved feature_importance.png")