"""File used to train a model for the UI"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.dataprocessing.data_processor import load_data, handle_extremal_values
from src.dataprocessing.text_cleaner import clean_text
from src.dataprocessing.split_vectorizer import tfidf_vectorizer
from src.defines import DATA_PATH, INDEPENDENT_FIELD, DEPENDENT_FIELD
from src.defines import INDEPENDENT_FIELD_CLEAN, INDEPENDENT_FIELD_LEN

def train_model_for_app():
    """Function load data and trains a Logistic Regression on it using TFIDF"""
    df = load_data(DATA_PATH)
    df[INDEPENDENT_FIELD_LEN] = df[INDEPENDENT_FIELD].apply(len)
    df = handle_extremal_values(df, column=INDEPENDENT_FIELD_LEN)
    df[INDEPENDENT_FIELD_CLEAN] = df[INDEPENDENT_FIELD].apply(clean_text)
    x_train, x_test, y_train, _ = train_test_split(df[INDEPENDENT_FIELD_CLEAN],
                                                   df[DEPENDENT_FIELD],
                                                   test_size=0.2,
                                                   random_state=42)
    x_train_tfidf, _, vectroizer = tfidf_vectorizer(x_train, x_test, return_vectorizer=True)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_tfidf, y_train)
    return model, vectroizer
