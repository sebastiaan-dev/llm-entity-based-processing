import joblib
import re
import string

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


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


def classify_new_question(question, vectorizer, selector, model, label_encoder):
    preprocessed = preprocess_text(question)
    question_tfidf = vectorizer.transform([preprocessed])
    question_selected = selector.transform(question_tfidf)
    predicted_label_encoded = model.predict(question_selected)[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_encoded])[0]
    return predicted_label


def test_model():
    svc_model = joblib.load("regression/best_svc_model.joblib")
    vectorizer = joblib.load("regression/tfidf_vectorizer.joblib")
    selector = joblib.load("regression/feature_selector.joblib")
    label_encoder = joblib.load("regression/label_encoder.joblib")

    new_question = "What currency is used in Japan?"
    predicted_label = classify_new_question(
        new_question, vectorizer, selector, svc_model, label_encoder
    )
    print(f"Predicted Label: {predicted_label}")


if __name__ == "__main__":
    test_model()
