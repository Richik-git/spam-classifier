import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# download required resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

# initialize tools
# ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


def transform_text(text: str) -> str:
    """
    Clean and preprocess text for model input
    """

    # convert to lowercase
    text = text.lower()

    # tokenize
    tokens = nltk.word_tokenize(text)

    cleaned_tokens = []

    for word in tokens:

        # keep only alphanumeric words
        if word.isalnum():

            # remove stopwords
            if word not in stop_words:

                # stemming
                # word = ps.stem(word)

                cleaned_tokens.append(word)

    return " ".join(cleaned_tokens)

if __name__ == "__main__":
    print(transform_text("Congratulations! You have won FREE money!!!"))