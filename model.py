import argparse
import os.path
from os.path import join

import pandas as pd

from src.runner import ExperimentRunner
from src.tuner import HyperparameterTuner
from src.utils import load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run feature selection and model evaluation.")
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--model-name", type=str, default="NSTransformer")
    parser.add_argument("--metric", type=str, help="Scoring metric", default="rmse")
    parser.add_argument("--horizon", type=int, help="horizon", default=7)
    parser.add_argument("--processes", "-p", type=int, default=4)
    parser.add_argument("--parameter-file", type=str, default="parameters.jsonl")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    snapshot_dt = pd.Timestamp('2024-03-01')
    base_path = os.getcwd()
    data = pd.read_pickle(
        "/Users/tony/Desktop/Project/tlab-causal-explorer/tlab-causal-explorer/input/gold_spot_price.pkl.bz2")
    ext_data = pd.read_pickle(
        "/Users/tony/Desktop/Project/tlab-causal-explorer/tlab-causal-explorer/input/gold_spot_external.pkl.bz2")
    parameter_path = join(base_path, args.parameter_file)
    parameter_dict = load_json(parameter_path)
    offset = 0
    horizon = 4
    dt = "20250124"

    target_column = "Com_Gold"
    model_list = ['nhits', 'nbeats', 'tsmixer', 'tide', 'rnn']
    output_folder = join(base_path, 'output', dt)

    # ExperimentRunner 초기화 (valid 모드)
    runner = ExperimentRunner(
        data=data,
        ext_data=ext_data,
        target_column=target_column,
        snapshot_dt=snapshot_dt,
        offset=offset,
        horizon=horizon,
        mode='valid',
        input_chunk_length=12,
        output_chunk_length=4,
        output_folder=output_folder,
        freq='W-MON',
        parameter_dict=parameter_dict,
        model_name=model_name
    )

    # 각 모델에 대해 하이퍼파라미터 튜닝 후, 최적 파라미터로 실험 실행
    tuned_results = {}
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
    for model, params in tuned_results.items():
        print(f"{model}: {params}")
