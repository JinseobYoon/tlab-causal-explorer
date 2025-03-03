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
    parameter_dict = load_json(parameter_path)
    parameter_dict['init_time'] = start_time

    select_method = parameter_dict['select_method']
    model_name = parameter_dict['model']
    target = parameter_dict['target']

    # Load and preprocess data
    data = pd.read_pickle(os.path.join(base_path, "input", "gold_spot_price.pkl.bz2"))
    data.dropna(axis=0, how='any', inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    # X, y = data.drop(target, axis=1), data[target]

    # Feature selection
    # selector = FeatureSelector(data=data,
    #                            target_col=target,
    #                            method=select_method)
    # features = selector.select_features()

    exp = ExpMain(parameter_dict)
    model = exp.train(setting=join(model_name, select_method, start_time))  # 학습 실행
    exp.test(setting=join(model_name, select_method, start_time))
