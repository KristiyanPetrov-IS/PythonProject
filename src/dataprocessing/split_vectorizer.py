"""File used to help with vectorization for independent data needed for ML models"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

def verctorize(x_train, x_test, vectorizer, return_vectorizer=False):
    """Apply vectorization, based on recieved vectorizer"""
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)
    if return_vectorizer:
        return x_train_vectorized, x_test_vectorized, vectorizer
    return x_train_vectorized, x_test_vectorized

def bow_vectorizer(x_train, x_test):
    """Create bow vectorization"""
    vectorizer = CountVectorizer()
    return verctorize(x_train, x_test, vectorizer)

def tfidf_vectorizer(x_train, x_test, return_vectorizer = False):
    """Create tfidf vectorization"""
    vectorizer = TfidfVectorizer()
    return verctorize(x_train, x_test, vectorizer, return_vectorizer)

def scale_max_abs(x_train, x_test):
    """Scale vectorized independent data"""
    scaler = MaxAbsScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled
