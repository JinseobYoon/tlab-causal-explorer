import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score


def load_data(random_state=42):
    np.random.seed(random_state)
    data = pd.DataFrame(
        np.random.rand(1000, 50), columns=[f"Feature_{i}" for i in range(50)]
    )
    data["Target"] = np.random.choice([0, 1], size=1000)
    return data


def evaluate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics = {
        "precision": precision,
        "accuracy": accuracy,
        "recall": recall,
        "f1": f1,
    }
    return metrics


def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    from sklearn.preprocessing import KBinsDiscretizer
    from statsmodels.tsa.stattools import adfuller
    from scipy.stats import kendalltau
    import mlflow

    """
    Preprocess the data by handling missing values and discretizing features.
    Also performs stationarity and nonlinearity testing.
    """
    # Drop missing values
    data = data.dropna()

    # Stationarity Testing
    print("Stationarity Testing with ADF:")
    stationarity_results = {}

    for column in data.columns[:-1]:
        adf_result = adfuller(data[column])
        stationarity_results[column] = {"ADF Statistic": adf_result[0], "p-value": adf_result[1]}
        print(f"{column}: ADF Statistic={adf_result[0]}, p-value={adf_result[1]}")

    mlflow.log_param("stationarity", stationarity_results)

    # Nonlinearity Testing (example using Kendall Tau)
    print("Nonlinearity Testing with Kendall Tau:")
    kendall_results = {}

    for column in data.columns[:-1]:
        tau, p_value = kendalltau(data[column], data['Target'])
        kendall_results[column] = {"ADF Kendall Tau": tau, "p-value": p_value}

        print(f"{column}: Kendall Tau={tau}, p-value={p_value}")
    mlflow.log_param("non-linearity", kendall_results)

    # Normalize data
    discretizer = KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="quantile")
    X = discretizer.fit_transform(data.drop(columns=["Target"]))
    y = data["Target"].values

    return pd.DataFrame(X, columns=data.columns[:-1]), y
