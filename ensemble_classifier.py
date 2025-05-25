import joblib
import numpy as np

class EnsembleColorClassifier:
    def __init__(self, rf_path, xgb_path, encoder_path, mode="hard"):
        self.rf = joblib.load(rf_path)
        self.xgb = joblib.load(xgb_path)
        self.encoder = joblib.load(encoder_path)
        self.mode = mode  # "hard" or "soft"

    def predict(self, X):
        if self.mode == "hard":
            return self._hard_vote(X)
        elif self.mode == "soft":
            return self._soft_vote(X)
        else:
            raise ValueError("Mode must be 'hard' or 'soft'")

    def _hard_vote(self, X):
        rf_pred = self.rf.predict(X)
        xgb_pred = self.xgb.predict(X)
        preds = []
        for i in range(len(X)):
            votes = [rf_pred[i], xgb_pred[i]]
            preds.append(max(set(votes), key=votes.count))
        return self.encoder.inverse_transform(preds)

    def _soft_vote(self, X):
        rf_proba = self.rf.predict_proba(X)
        xgb_proba = self.xgb.predict_proba(X)
        avg_proba = (rf_proba + xgb_proba) / 2
        soft_preds = np.argmax(avg_proba, axis=1)
        return self.encoder.inverse_transform(soft_preds)
