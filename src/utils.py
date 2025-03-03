import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score


def load_json(base_path):
    import json
    # JSON 파일 불러오기
    with open(base_path, "r") as f:
        loaded_config = json.load(f)

    return loaded_config


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


def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
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
    y = data["Target"]

    return pd.DataFrame(X, columns=data.columns[:-1]), y


### Masking 함수 정의
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


### Learning_rate 학습 함수 정의
def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


### EarlyStopping 함수 정의
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


### 시각화 함수 정의
def visual(true, preds=None, name='./pic/test.pdf'):
    import matplotlib.pyplot as plt
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


### 평가지표 함수 정의
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
