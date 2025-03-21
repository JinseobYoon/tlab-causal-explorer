import os
import pandas as pd
from src.feature_select import FeatureSelector

def load_data():
    base_path = os.getcwd()
    file_path = os.path.join(base_path, "input", "gold_spot_price.pkl.bz2")
    data = pd.read_pickle(file_path)
    data.dropna(axis=0, how='any', inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')

    if 'dt' in data.columns:
        data.drop(columns=['dt'], inplace=True)
    return data

if __name__ == "__main__":
    data = load_data()
    target = "Com_Gold"
    methods = ["NBCB"]

    for method in methods:
        selector = FeatureSelector(data=data, target_col=target, method=method)
        try:
            result = selector.select_features()

            if method == "NBCB":
                full_result, com_gold_causes = result
                print("Com_Gold에 영향을 미치는 변수:", com_gold_causes)

        except Exception as e:
            print(f"Error occurred while testing method {method}: {e}")