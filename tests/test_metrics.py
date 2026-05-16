from sklearn.tree import DecisionTreeClassifier

from hib.metrics import evaluate_model
from hib.synthetic import SyntheticSkewConfig, make_train_test_split


def test_metrics_compute_successfully():
    config = SyntheticSkewConfig(skew_ratio=1, seed=7, minority_count=20)
    X_train, X_test, y_train, y_test = make_train_test_split(config)
    model = DecisionTreeClassifier(random_state=7).fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    assert set(metrics) == {
        "auroc",
        "average_precision",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
        "brier_score",
    }
    assert all(isinstance(value, float) for value in metrics.values())
