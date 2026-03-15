import pickle
from preprocess import transform_text


def load_model():

    with open("models/spam_pipeline.pkl", "rb") as f:
        model = pickle.load(f)

    return model


def predict_message(message):

    model = load_model()

    cleaned = transform_text(message)

    result = model.predict([cleaned])[0]

    return "Spam" if result == 1 else "Ham"


if __name__ == "__main__":

    msg = input("Enter message: ")

    prediction = predict_message(msg)

    print("Prediction:", prediction)