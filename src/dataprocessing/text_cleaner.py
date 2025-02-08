"""File to add needed fields to the dataframe, based on the existing fields"""
import re
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.defines import INDEPENDENT_FIELD, INDEPENDENT_FIELD_CLEAN
from src.defines import INDEPENDENT_FIELD_SENTIMENT, INDEPENDENT_FIELD_TOKEN
from src.defines import INDEPENDENT_FIELD_LEMMAT, INDEPENDENT_FIELD_STEM

def clean_text(text):
    """Remove whitespace and non num/char"""
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def lemmatize_text(text):
    """Apply lemmatization"""
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])

def stem_text(text):
    """Apply stemming"""
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in word_tokenize(text)])

def add_processed_fields(df, has_lemmatization = False):
    """Process fields and add clean_review, sentiment, token. Option add lemmat and stemm fileds"""
    df[INDEPENDENT_FIELD_CLEAN] = df[INDEPENDENT_FIELD].apply(clean_text)
    df[INDEPENDENT_FIELD_SENTIMENT] = df[INDEPENDENT_FIELD].apply(
        lambda x: TextBlob(x).sentiment.polarity)

    # Disable by default, with control to enable as these take runtime
    if has_lemmatization:
        df[INDEPENDENT_FIELD_LEMMAT] = df[INDEPENDENT_FIELD_CLEAN].apply(lemmatize_text)
        df[INDEPENDENT_FIELD_STEM] = df[INDEPENDENT_FIELD_CLEAN].apply(stem_text)
        df[INDEPENDENT_FIELD_TOKEN] = df[INDEPENDENT_FIELD_LEMMAT].apply(word_tokenize)
    else:
        df[INDEPENDENT_FIELD_TOKEN] = df[INDEPENDENT_FIELD_CLEAN].apply(word_tokenize)
    return df
