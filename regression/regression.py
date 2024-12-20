import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
# df = pd.read_csv("regression/questions_dataset.csv")
df = pd.read_csv("data_synthetic/data_synthetic.csv")

# Display the first few rows
print(df.head())


# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the labels
# df["label_encoded"] = label_encoder.fit_transform(df["Category0"])
df["label_encoded"] = label_encoder.fit_transform(df["Entity-Type"])

# Mapping of labels to numerical values
label_mapping = dict(
    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
)
print("Label Mapping:", label_mapping)


# Features and target
X = df["Question"]
y = df["label_encoded"]

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    sublinear_tf=True,  # Apply sublinear TF scaling
    max_features=10000,  # Limit to top 5000 features
    ngram_range=(1, 3),  # Use unigrams and bigrams
    stop_words="english",  # Remove common English stop words
)

# Fit the vectorizer on training data and transform both training and testing data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF Training Matrix Shape: {X_train_tfidf.shape}")
print(f"TF-IDF Testing Matrix Shape: {X_test_tfidf.shape}")


# Initialize Logistic Regression model
log_reg = LogisticRegression(
    multi_class="multinomial",  # Suitable for multi-class classification
    solver="lbfgs",  # Solver for optimization
    max_iter=2000,  # Maximum number of iterations
    random_state=42,
)

# Train the model
log_reg.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred_log_reg = log_reg.predict(X_test_tfidf)

# Evaluate the model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred_log_reg, target_names=label_encoder.classes_),
)

new_question = "Where is the author of the orginal book?"
preprocessed_question = new_question.lower()
question_tfidf = vectorizer.transform([preprocessed_question])
predicted_label_encoded = log_reg.predict(question_tfidf)[0]
predicted_label = label_encoder.inverse_transform([predicted_label_encoded])[0]
print(f"Predicted Label: {predicted_label}")

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform

# Define the parameter distribution
param_dist = {
    "C": loguniform(1e-3, 1e3),
    "gamma": loguniform(1e-4, 1e1),
    "kernel": ["rbf", "linear", "poly", "sigmoid"],
    "degree": [2, 3, 4, 5],  # Relevant for 'poly' kernel
}

# Initialize SVC
svc = SVC()

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=svc,
    param_distributions=param_dist,
    n_iter=10,  # Number of parameter settings sampled
    scoring="accuracy",
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

# Fit RandomizedSearchCV
random_search.fit(X_train_tfidf, y_train)

# Best parameters and score
print("Best Parameters from Randomized Search:", random_search.best_params_)
print("Best Cross-Validation Accuracy:", random_search.best_score_)

# Create a baseline classifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)

# Predict and evaluate baseline accuracy
y_pred = dummy_clf.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred)
print(f"Baseline Accuracy (Most Frequent): {baseline_accuracy:.4f}")
