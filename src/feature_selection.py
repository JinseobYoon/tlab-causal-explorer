import pandas as pd
from pgmpy.estimators import PC


def _select_features_cfs(X, y):
    data = pd.concat([X, y], axis=1)
    estimator = PC(data=data)
    skeleton = estimator.estimate()
    causal_features = [f for f in X.columns if skeleton.get_neighbors(f)]
    return causal_features


def _select_features_correlation(X, y, threshold=0.1):
    correlations = X.corrwith(y)
    return correlations[correlations.abs() > threshold].index.tolist()


class FeatureSelector:
    def __init__(self, method="CFS"):
        self.method = method

    def select_features(self, X: pd.DataFrame, y: pd.Series):
        if self.method == "CFS":
            return _select_features_cfs(X, y)
        elif self.method == "Correlation":
            return _select_features_correlation(X, y)
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")
