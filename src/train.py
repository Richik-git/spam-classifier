import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from preprocess import transform_text


def load_data(filepath):
    """
    Load dataset and clean basic columns
    """

    df = pd.read_csv(filepath, encoding="latin-1")

    # keep only useful columns
    df = df[["v1", "v2"]]

    # rename columns
    df.columns = ["target", "text"]

    return df


def preprocess_data(df):
    """
    Apply label encoding and text preprocessing
    """

    # encode labels
    encoder = LabelEncoder()
    df["target"] = encoder.fit_transform(df["target"])

    # remove duplicates
    df = df.drop_duplicates(keep="first")

    # apply text preprocessing
    df["clean_text"] = df["text"].apply(transform_text)

    return df

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score

def train_model(df):
    """
    Train spam classification model using pipeline
    """

    X = df["clean_text"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=2,
        stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2)
        )),
        ("model", MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))

    return pipeline


def save_model(model):

    with open("models/spam_pipeline.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":

    df = load_data("data/spam.csv")
    df = preprocess_data(df)
    pipeline = train_model(df)
    save_model(pipeline)

    print("Pipeline saved successfully!")