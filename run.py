import argparse
import datetime
import os.path
from os.path import join

import pandas as pd

from src.exp import ExpMain
from src.utils import load_json


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

    ## 수정
    exp = ExpMain(base_config)
    model = exp.train(setting=join(model_name, select_method, start_time))  # 학습 실행
    exp.test(setting=join(model_name, select_method, start_time))



