import pandas as pd
import numpy as np
import string
import re
from sklearn.base import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
import joblib

# import matplotlib.pyplot as plt
# import seaborn as sns
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, download

# Download necessary NLTK data
download("punkt")
download("wordnet")

# Load the dataset
df = pd.read_csv("data_synthetic/data_synthetic.csv")

# Display the first few rows
print(df.head())

# Text Preprocessing Function
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", "", text)
    # Tokenize
    tokens = word_tokenize(text)
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


df["Question"] = df["Question"].apply(preprocess_text)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the labels
df["label_encoded"] = label_encoder.fit_transform(df["Entity-Type"])

# Mapping of labels to numerical values
label_mapping = dict(
    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
)
print("Label Mapping:", label_mapping)

# Features and target
X = df["Question"]
y = df["label_encoded"]

# Split the data (80% training, 20% testing) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=10000,  # Increased feature space
    ngram_range=(1, 3),  # Included trigrams
    stop_words="english",
    sublinear_tf=True,
)

# Fit the vectorizer on training data and transform both training and testing data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF Training Matrix Shape: {X_train_tfidf.shape}")
print(f"TF-IDF Testing Matrix Shape: {X_test_tfidf.shape}")

# Feature Selection
selector = SelectKBest(chi2, k=5000)
X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
X_test_selected = selector.transform(X_test_tfidf)

print(f"Selected Training Features Shape: {X_train_selected.shape}")
print(f"Selected Testing Features Shape: {X_test_selected.shape}")

# Hyperparameter Tuning with Grid Search
param_grid = {
    "C": [1, 10, 100, 1000],
    "gamma": [0.001, 0.01, 0.1, 1, "scale", "auto"],
    "kernel": ["rbf", "linear", "poly", "sigmoid"],
}

grid_search = GridSearchCV(
    estimator=SVC(class_weight="balanced", probability=True, random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    verbose=2,
    n_jobs=-1,
)

# Fit GridSearchCV
grid_search.fit(X_train_selected, y_train)

# Best parameters and score
print("Best Parameters from Grid Search:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate the best model on the test set
best_svc = grid_search.best_estimator_
y_pred_best_svc = best_svc.predict(X_test_selected)
print("Best SVC Accuracy on Test Set:", accuracy_score(y_test, y_pred_best_svc))
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred_best_svc, target_names=label_encoder.classes_),
)

# Save the trained models
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
joblib.dump(selector, "feature_selector.joblib")
joblib.dump(best_svc, "best_svc_model.joblib")
joblib.dump(label_encoder, "label_encoder.joblib")


# Function to classify new questions
def classify_new_question(question, vectorizer, selector, model, label_encoder):
    preprocessed = preprocess_text(question)
    question_tfidf = vectorizer.transform([preprocessed])
    question_selected = selector.transform(question_tfidf)
    predicted_label_encoded = model.predict(question_selected)[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_encoded])[0]
    return predicted_label


# Example Usage
new_question = "Where is the author of the original book?"
predicted_label = classify_new_question(
    new_question, vectorizer, selector, best_svc, label_encoder
)
print(f"Predicted Label: {predicted_label}")
