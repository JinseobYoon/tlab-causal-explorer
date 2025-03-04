import argparse
import datetime
import os.path
from os.path import join

import pandas as pd

from src.exp import ExpMain
from src.utils import load_json
from src.tuner import HyperParameterTuner

def parse_args():
    parser = argparse.ArgumentParser(description="Run feature selection and model evaluation.")
    parser.add_argument("--parameter-file", type=str, default="parameters.jsonl")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")

    base_path = os.getcwd()
    parameter_path = join(base_path, args.parameter_file)
    base_config = load_json(parameter_path)
    base_config['init_time'] = start_time

    select_method = base_config['select_method']
    model_name = base_config['model']
    target = base_config['target']

    # Load and preprocess data
    data = pd.read_pickle(os.path.join(base_path, "input", "gold_spot_price.pkl.bz2"))
    data.dropna(axis=0, how='any', inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')


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


    # exp = ExpMain(base_config)
    # model = exp.train(setting=join(model_name, select_method, start_time))  # 학습 실행
    # exp.test(setting=join(model_name, select_method, start_time))