import joblib


def preprocess_question(q: str) -> str:
    return q.lower().strip()


def classify_new_question(question, model, label_encoder):
    preprocessed = preprocess_question(question)
    pred_encoded = model.predict([preprocessed])
    pred_label = label_encoder.inverse_transform(pred_encoded)[0]

    return pred_label


def test_model():
    svc_model = joblib.load("regression/best_model.joblib")
    label_encoder = joblib.load("regression/label_encoder.joblib")

    new_question = "Where is the Great Barrier Reef located?"
    predicted_label = classify_new_question(new_question, svc_model, label_encoder)
    print(f"Predicted Label: {predicted_label}")


if __name__ == "__main__":
    test_model()
