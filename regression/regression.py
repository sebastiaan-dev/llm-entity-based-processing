import joblib
import pandas as pd

from sklearn.calibration import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint, uniform, loguniform


def preprocess_question(q: str) -> str:
    return q.lower().strip()


def prepare_data():
    label_encoder = LabelEncoder()
    df = pd.read_csv("data_synthetic/data_synthetic.csv")

    df["Question"] = df["Question"].apply(preprocess_question)
    df["label_encoded"] = label_encoder.fit_transform(df["Entity-Type"])

    X = df["Question"]
    y = df["label_encoded"]

    return X, y, label_encoder


def train_model(X, y):
    # https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
    # https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html
    base_clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)

    # https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.BaggingClassifier.html
    bagging_clf = BaggingClassifier(
        estimator=base_clf,
        n_estimators=10,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42069,
    )
    # https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=10000)),
            ("svd", TruncatedSVD(n_components=300, random_state=42069)),
            ("scaler", StandardScaler(with_mean=False)),
            ("bagging", bagging_clf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42069, stratify=y
    )

    param_dist = {
        "bagging__n_estimators": randint(10, 50),
        "bagging__max_samples": uniform(0.6, 0.4),
        "bagging__max_features": uniform(0.6, 0.4),
        "bagging__estimator__C": loguniform(1e-2, 1e3),
        "bagging__estimator__solver": [
            "lbfgs",
            # "saga",
        ],
    }

    # https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=30,
        cv=5,
        scoring="accuracy",
        random_state=42069,
        n_jobs=6,
        verbose=2,
    )

    random_search.fit(X_train, y_train)

    # Predict on the test set with the best estimator
    y_pred_best = random_search.predict(X_test)

    # Evaluate the best model
    print("Test Accuracy:", accuracy_score(y_test, y_pred_best))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

    return random_search


def main():
    X, y, label_encoder = prepare_data()
    model = train_model(X, y)

    # https://www.analyticsvidhya.com/blog/2023/02/how-to-save-and-load-machine-learning-models-in-python-using-joblib-library/
    joblib.dump(label_encoder, "label_encoder.joblib")
    joblib.dump(model.best_estimator_, "best_model.joblib")


if __name__ == "__main__":
    main()
