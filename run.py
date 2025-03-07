import argparse
import json
import os.path

import pandas as pd

import wandb
from src.exp import ExpMain


def parse_args():
    parser = argparse.ArgumentParser(description="Run feature selection and model evaluation.")
    parser.add_argument("--parameter-file", type=str, default="parameters.jsonl")
    return parser.parse_args()


# ✅ Wandb Sweep을 실행할 train 함수
def train_sweep():
    wandb.init()  # Wandb 실행

    # Wandb에서 설정된 하이퍼파라미터 가져오기
    config = wandb.config

    # 기존 base_config 불러오기
    args = parse_args()
    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    base_path = os.getcwd()
    parameter_path = join(base_path, args.parameter_file)
    base_config = load_json(parameter_path)
    base_config['init_time'] = start_time

    # ✅ Wandb 설정값으로 base_config 업데이트
    base_config.update({
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "model_params": {
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "dropout": config.dropout,
            "activation": config.activation
        }
    })

    select_method = base_config['select_method']
    model_name = base_config['model']

    # ✅ ExpMain 실행
    exp = ExpMain(base_config)
    model = exp.train(setting=join(model_name, select_method, start_time))
    exp.test(setting=join(model_name, select_method, start_time))
    wandb.finish()


if __name__ == '__main__':
    import datetime
    import os
    from src.utils import load_json
    from src.feature_select import FeatureSelector
    from os.path import join

    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")

    base_path = os.getcwd()
    parameter_path = join("../", base_path, "parameters.jsonl")
    base_config = load_json(parameter_path)
    base_config['init_time'] = start_time

    select_method = base_config['select_method']
    model_name = base_config['model']
    target = base_config['target']

    # Load and preprocess data
    data = pd.read_pickle(os.path.join(base_path, "input", "gold_spot_price.pkl.bz2"))
    data.dropna(axis=0, how='any', inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    selector = FeatureSelector(data=data, target_col=target, method=select_method)
    feature_set = selector.select_features()

    # ✅ Sweep 설정 파일에서 불러오기
    with open("sweep_config.jsonl", "r") as f:
        sweep_config = json.load(f)

    # ✅ Sweep 등록
    sweep_id = wandb.sweep(sweep_config, project="NSTransformer_Experiments")

    # ✅ Sweep 실행
    wandb.agent(sweep_id, function=train_sweep, count=10)
