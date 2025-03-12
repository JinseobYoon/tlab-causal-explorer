import os
import pandas as pd
from src.feature_select import FeatureSelector

def load_data():
    base_path = os.getcwd()
    file_path = os.path.join(base_path, "input", "gold_spot_price.pkl.bz2")
    data = pd.read_pickle(file_path)
    data.dropna(axis=0, how='any', inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    return data

if __name__ == "__main__":
    print("Test script started!")

    # 데이터 로드
    data = load_data()
    print("데이터 로드 완료. 데이터 shape:", data.shape)
    print("데이터 컬럼 목록:", data.columns.tolist())

    target = "Com_Gold"
    methods = ["NBCB"]

    for method in methods:
        print(f"\n=== Testing Feature Selection Method: {method} ===")
        selector = FeatureSelector(data=data, target_col=target, method=method)
        try:
            result = selector.select_features()
            if isinstance(result, pd.DataFrame):
                print(result.head())
            else:
                print(result)
        except Exception as e:
            print(f"Error occurred while testing method {method}: {e}")
