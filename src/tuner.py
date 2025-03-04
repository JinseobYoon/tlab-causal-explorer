import copy
from types import SimpleNamespace

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit

# 필요한 모듈들을 import합니다.
from src.exp import ExpMain  # 기존의 학습/평가를 담당하는 Trainer 클래스
from src.runner import data_provider


class HyperParameterTuner:
    def __init__(self, base_config, param_ranges, n_splits=3, n_trials=50):
        """
        Parameters:
            base_config (dict): 고정 설정 값 (데이터, 모델 기본 설정 등)
            param_ranges (dict): 튜닝할 하이퍼파라미터의 범위 및 타입 정보
            n_splits (int): TS‑CV에서 사용할 fold 수
            n_trials (int): Optuna에서 시도할 trial 수
        """
        self.base_config = base_config
        self.param_ranges = param_ranges
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.best_params = None

    def _train_and_evaluate(self, args, setting="trial_experiment"):
        """
        ExpMain 객체를 생성하여 학습을 진행한 후, validation 데이터에 대한 loss(MSE)를 반환합니다.
        """
        trainer = ExpMain(args)
        trainer.train(setting=setting)
        vali_data, vali_loader = trainer._get_data(flag='val')
        criterion = trainer._select_criterion()
        val_loss = trainer.vali(vali_data, vali_loader, criterion)
        return val_loss

    def _run_trial(self, args):
        """
        TimeSeriesSplit을 이용하여, 각 fold마다 모델을 학습/평가하고,
        fold별 validation loss의 평균을 반환합니다.

        주의: data_provider 함수 또는 ExpMain 클래스가 args 내에
              train_indices, val_indices를 활용하여 해당 인덱스의 데이터만 반환하도록 구현되어 있어야 합니다.
        """
        # 전체 데이터셋 불러오기 (train flag)
        dataset, _ = data_provider(args, flag="train")
        total_indices = np.arange(len(dataset.data_x))
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_losses = []
        fold_id = 0

        for train_idx, val_idx in tscv.split(total_indices):
            fold_id += 1
            print(f"--- TS-CV Fold {fold_id}/{self.n_splits} ---")
            # args를 복제하고, fold별 train/val 인덱스 정보를 추가
            fold_args = copy.deepcopy(args)
            fold_args.train_indices = train_idx.tolist()
            fold_args.val_indices = val_idx.tolist()

            fold_loss = self._train_and_evaluate(fold_args, setting=f"trial_fold_{fold_id}")
            print(f"Fold {fold_id} validation loss: {fold_loss:.4f}")
            fold_losses.append(fold_loss)

        avg_loss = np.mean(fold_losses)
        print(f"Average validation loss over {self.n_splits} folds: {avg_loss:.4f}")
        return avg_loss

    def objective(self, trial):
        """
        Optuna Objective 함수.
        샘플링된 하이퍼파라미터를 base_config와 병합하여 TS‑CV 평가를 진행하고,
        평균 validation loss를 반환합니다.
        """
        sampled_params = {}
        for param, config in self.param_ranges.items():
            if config.get("type") == "int":
                step = config.get("step", 1)
                sampled_params[param] = trial.suggest_int(param, config["low"], config["high"], step=step)
            elif config.get("type") == "float":
                if config.get("log", False):
                    sampled_params[param] = trial.suggest_float(param, config["low"], config["high"], log=True)
                else:
                    sampled_params[param] = trial.suggest_float(param, config["low"], config["high"])
            elif config.get("type") == "categorical":
                sampled_params[param] = trial.suggest_categorical(param, config["choices"])

        # base_config와 sampled_params 병합
        config = self.base_config.copy()
        config.update(sampled_params)
        args = SimpleNamespace(**config)

        # TS‑CV 기반 평가 실행
        val_loss = self._run_trial(args)
        return val_loss

    def run_study(self):
        """
        Optuna Study를 실행하여 최적의 하이퍼파라미터를 탐색합니다.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        print("Best hyperparameters:", study.best_params)
        self.best_params = study.best_params
        return study

    def train_final_model(self):
        """
        최적의 하이퍼파라미터(best_params)를 바탕으로 전체 training set으로 모델을 재학습하고,
        test set에 대해 최종 평가를 진행합니다.
        """
        final_config = self.base_config.copy()
        final_config.update(self.best_params)
        final_args = SimpleNamespace(**final_config)

        trainer = ExpMain(final_args)
        trainer.train(setting="final_experiment")
        trainer.test(setting="final_experiment", test=0)
        print("Final model training and testing complete.")


# ------------------------------------------------
# 사용 예시
# ------------------------------------------------
if __name__ == "__main__":
    # 고정 설정값 (데이터, 모델, 기타 설정)
    base_config = {
        "data": "custom",
        "model": "NSTransformer",
        "root_path": "/Users/tony/Desktop/Project/tlab-causal-explorer/tlab-causal-explorer/input/",
        "data_path": "gold_spot_price.pkl.bz2",
        "features": "MS",
        "target": "Com_Gold",
        "scale": False,
        "freq": "b",
        "checkpoints": "./checkpoints/",
        "seq_len": 90,
        "label_len": 45,
        "pred_len": 90,
        "enc_in": 37,
        "dec_in": 37,
        "c_out": 1,
        "num_workers": 0,
        "train_epochs": 10,
        "patience": 3,
        "loss": "mse",
        "lradj": "type1",
        "use_amp": False,
        "use_gpu": False,
        "gpu": 0,
        "use_multi_gpu": False,
        "output_attention": False,
        "embed": "fixed",  # 추가: embed 항목
        "factor": 1,

    }

    # 하이퍼파라미터 범위 정보
    param_ranges = {
        "d_model": {"type": "int", "low": 128, "high": 1024, "step": 128},
        "n_heads": {"type": "int", "low": 2, "high": 16, "step": 2},
        "e_layers": {"type": "int", "low": 1, "high": 4},
        "d_layers": {"type": "int", "low": 1, "high": 3},
        "d_ff": {"type": "int", "low": 512, "high": 4096, "step": 512},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
        "activation": {"type": "categorical", "choices": ["relu", "gelu"]},
        "p_hidden_dims": {"type": "categorical", "choices": [[16, 16], [32, 32], [64, 64]]},
        "p_hidden_layers": {"type": "int", "low": 1, "high": 4}
    }

    tuner = HyperParameterTuner(base_config, param_ranges, n_splits=3, n_trials=50)

    # 하이퍼파라미터 튜닝 실행 (TS-CV 기반)
    study = tuner.run_study()

    # 최적 파라미터를 사용하여 최종 모델 학습 및 테스트
    tuner.train_final_model()
