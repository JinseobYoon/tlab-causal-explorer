import numpy as np
import optuna

from src.runner import ExperimentRunner


# --- Hyperparameter Tuner ---
class HyperparameterTuner:
    @staticmethod
    def get_model_params(model_name: str, trial: optuna.trial.Trial):
        if model_name == 'dlinear':
            return {
                "kernel_size": trial.suggest_int("kernel_size", 3, 7),
                "n_epochs": trial.suggest_int("n_epochs", 50, 150)
            }
        elif model_name == 'nhits':
            return {
                "num_stacks": trial.suggest_int("num_stacks", 2, 4),
                "num_blocks": trial.suggest_int("num_blocks", 1, 3),
                "num_layers": trial.suggest_int("num_layers", 2, 4),
                "layer_widths": trial.suggest_int("layer_widths", 128, 512),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5)
            }
        elif model_name == 'nbeats':
            return {
                "num_stacks": trial.suggest_int("num_stacks", 2, 4),
                "num_blocks": trial.suggest_int("num_blocks", 1, 3),
                "num_layers": trial.suggest_int("num_layers", 2, 4),
                "layer_widths": trial.suggest_int("layer_widths", 128, 512)
            }
        elif model_name == 'nlinear':
            return {}  # 추가 하이퍼파라미터 없음
        elif model_name == 'tcn':
            return {
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "dilation_base": trial.suggest_int("dilation_base", 2, 4),
                "kernel_size": trial.suggest_int("kernel_size", 2, 5),
                "num_filters": trial.suggest_int("num_filters", 16, 64)
            }
        elif model_name == 'tsmixer':
            return {
                "hidden_size": trial.suggest_int("hidden_size", 32, 128),
                "ff_size": trial.suggest_int("ff_size", 32, 128),
                "num_blocks": trial.suggest_int("num_blocks", 1, 4),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5)
            }
        elif model_name == 'tide':
            return {
                "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 3),
                "num_decoder_layers": trial.suggest_int("num_decoder_layers", 1, 3),
                "decoder_output_dim": trial.suggest_int("decoder_output_dim", 16, 64),
                "hidden_size": trial.suggest_int("hidden_size", 64, 256)
            }
        elif model_name in ['rnn', 'gru', 'lstm']:
            return {
                "hidden_dim": trial.suggest_int("hidden_dim", 32, 128),
                "n_rnn_layers": trial.suggest_int("n_rnn_layers", 1, 5)
            }
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    @staticmethod
    def objective(trial: optuna.trial.Trial, model_name: str, runner: ExperimentRunner):
        # 모델별 하이퍼파라미터 획득
        model_params = HyperparameterTuner.get_model_params(model_name, trial)
        # 실험 실행 (valid 모드에서 예측 결과 DataFrame 획득)
        predictions_df, output_file = runner.run_model(model_name, model_params)
        if 'v' not in predictions_df.columns or predictions_df.empty:
            return float("inf")
        # sMAPE 계산 (낮을수록 좋음)
        smape_list = []
        for idx, row in predictions_df.iterrows():
            actual = row['v']
            pred = row['pred']
            if (abs(actual) + abs(pred)) == 0:
                continue
            smape_val = 200 * abs(actual - pred) / (abs(actual) + abs(pred))
            smape_list.append(smape_val)
        return np.mean(smape_list) if smape_list else float("inf")

    @staticmethod
    def tune(model_name: str, runner: ExperimentRunner, n_trials: int = 50):
        study = optuna.create_study(direction="minimize", study_name=f"{model_name}_tuning")
        study.optimize(lambda trial: HyperparameterTuner.objective(trial, model_name, runner), n_trials=n_trials)
        print(f"Best trial for {model_name}:")
        print(f"  sMAPE: {study.best_trial.value}")
        print(f"  Params: {study.best_trial.params}")
        return study.best_trial
