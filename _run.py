import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.feature_select import FeatureSelector
from src.utils import load_data, preprocess_data, evaluate_metrics


# TODO 실험의 주 목적이 무엇인지 정의하고
def parse_args():
    parser = argparse.ArgumentParser(description="Run feature selection and model evaluation.")
    parser.add_argument("--feature_selection_method", type=str, default="NGC",
                        choices=["CFS", "Correlation"], help="Feature selection method to use.")
    parser.add_argument("--model_name", type=str, default="RandomForest",
                        choices=["RandomForest",
                                 "LogisticRegression",
                                 "SVM",
                                 "LSTM",
                                 "GRU",
                                 "Transformer",
                                 ], help="Model to train.")


    return parser.parse_args()


# --- Experiment Runner ---
class ExperimentRunner:
    def __init__(self, data: pd.DataFrame, snapshot_dt: pd.Timestamp, offset: int, horizon: int,
                 mode: str = 'valid', input_chunk_length: int = 12, output_chunk_length: int = 6,
                 output_folder: str = './results', freq: str = 'MS'):
        self.data = data
        self.snapshot_dt = snapshot_dt
        self.offset = offset
        self.horizon = horizon
        self.mode = mode
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.output_folder = output_folder
        self.freq = freq
        self.scaler = Scaler()

        # 날짜 범위 계산
        self.test_start = snapshot_dt - pd.DateOffset(months=offset)
        self.test_end = self.test_start + pd.DateOffset(months=horizon + offset)
        self.valid_start = self.test_start - pd.DateOffset(months=horizon + offset)
        self.valid_end = self.test_start - pd.DateOffset(months=1)
        print(f"valid_start: {self.valid_start}, \
                valid_end: {self.valid_end}, \
                test_start: {self.test_start}, \
                test_end: {self.test_end}")

    def run_model(self, model_name: str, model_params: dict = None):
        os.makedirs(self.output_folder, exist_ok=True)
        predictions = []

        for grain_id, group in self.data.groupby('grain_id'):
            if self.mode == "test":
                filled_group = DataPreparator.prepare_data(group, grain_id, self.test_start, self.test_end, self.freq)
            else:
                filled_group = group.copy()

            # 데이터 분할: Train, Validation, Test
            train = filled_group[filled_group['dt'] < self.valid_start]
            valid = filled_group[(filled_group['dt'] >= self.valid_start) & (filled_group['dt'] <= self.valid_end)]
            test = filled_group[(filled_group['dt'] >= self.test_start) & (filled_group['dt'] <= self.test_end)]

            if self.mode == 'test':
                train = pd.concat([train, valid], ignore_index=True)
            elif self.mode == 'valid':
                test = valid.copy()

            # TimeSeries 객체 생성 및 스케일링
            # train_series = TimeSeries.from_dataframe(train, time_col='dt', value_cols='v', freq=self.freq)
            # test_series = TimeSeries.from_dataframe(test, time_col='dt', value_cols='v', freq=self.freq)
            train_series = TimeSeries.from_dataframe(train, time_col='dt', value_cols=['v', 'kcd_v'], freq=self.freq)
            test_series = TimeSeries.from_dataframe(test, time_col='dt', value_cols=['v', 'ubist_v'], freq=self.freq)

            train_series = self.scaler.fit_transform(train_series)
            print(train_series)

            model = ModelFactory.create_model(model_name, self.input_chunk_length, self.output_chunk_length,
                                              model_params)
            model.fit(train_series)
            prediction_series = model.predict(len(test_series))
            prediction_series = self.scaler.inverse_transform(prediction_series)

            for i, dt in enumerate(test_series.time_index):
                record = {'grain_id': grain_id, 'dt': dt, 'pred': prediction_series.values()[i][0]}
                if self.mode == 'valid':
                    record['v'] = test_series.values()[i][0]
                    record['model'] = model_name
                else:
                    record['snapshot_dt'] = self.snapshot_dt
                predictions.append(record)

        predictions_df = pd.DataFrame(predictions)
        if self.mode == 'valid':
            output_file = join(self.output_folder, f'predictions_monthly_{model_name}_valid_v2.csv')
        elif self.mode == 'test':
            output_file = join(self.output_folder, f'predictions_monthly_{model_name}_next_v2.csv')
        else:
            output_file = join(self.output_folder, f'predictions_monthly_{model_name}_v2.csv')
        return predictions_df, output_file


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


def main():
    args = parse_args()

    runner = ExperimentRunner(
        data=data,
        ext_data=ext_data,
        snapshot_dt=snapshot_dt,
        offset=offset,
        horizon=horizon,
        mode='valid',
        input_chunk_length=12,
        output_chunk_length=4,
        output_folder=output_folder,
        freq='MS'
    )

    # Load and preprocess data
    data = load_data(random_state=args.random_state)
    X, y = preprocess_data(data)

    # Feature selection
    selector = FeatureSelector(X= x,
                               y= y,
                               method=args.feature_selection_method)
    features = selector.select_features(X, y)

    for model_name in model_list:
        print(f"\nTuning hyperparameters for model: {model_name}")
        best_trial = HyperparameterTuner.tune(model_name, runner, n_trials=50)
        tuned_results[model_name] = best_trial.params

        # 최적 파라미터로 최종 실험 실행 (예측 결과 저장)
        predictions, output_file = runner.run_model(model_name, model_params=best_trial.params)
        # 필요에 따라 predictions를 후처리하거나 저장할 수 있습니다.
        predictions.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    print("\nTuned hyperparameters for all models:")


    # Log metrics
    # TODO 관련 산출물 정의
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
