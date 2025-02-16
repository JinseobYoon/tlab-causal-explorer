import argparse

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.feature_selection import FeatureSelector
from src.utils import load_data, preprocess_data, evaluate_metrics

#TODO 실험의 주 목적이 무엇인지 정의하고
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
    parser.add_argument("--test_size", type=float, default=0.2, help="Test dataset size as a fraction.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Start MLflow experiment
    mlflow.set_experiment("Feature Selection and Model Evaluation")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("feature_selection_method", args.feature_selection_method)
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        # Load and preprocess data
        data = load_data(random_state=args.random_state)
        X, y = preprocess_data(data)

        # Feature selection
        selector = FeatureSelector(method=args.feature_selection_method)
        features = selector.select_features(X, y)

        # Log selected features
        mlflow.log_param("selected_features", features)

        # Train and evaluate the model
        models = {
            "RandomForest": RandomForestClassifier(random_state=args.random_state),
            "LogisticRegression": LogisticRegression(random_state=args.random_state),
            "SVM": SVC(random_state=args.random_state),
        }

        model = models[args.model_name]
        model.fit(X[features], y)
        y_pred = model.predict(X[features])

        # Evaluate metrics
        metrics = evaluate_metrics(y, y_pred)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, "model")
        # TODO 관련 산출물 정의
        print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
