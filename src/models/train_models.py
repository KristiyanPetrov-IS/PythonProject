"""File used to train and evaluate logistic regression and 
naive bayes modelson scaled and unscaled bow and tfidf"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.dataprocessing.split_vectorizer import bow_vectorizer, tfidf_vectorizer, scale_max_abs

def train_models_and_compare(x, y):
    """Trains and compares the models stated in the docstring"""

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train_bow, x_test_bow = bow_vectorizer(x_train, x_test)
    x_train_tfidf, x_test_tfidf = tfidf_vectorizer(x_train, x_test)

    x_train_bow_scaled, x_test_bow_scaled = scale_max_abs(x_train_bow, x_test_bow)
    x_train_tfidf_scaled, x_test_tfidf_scaled = scale_max_abs(x_train_tfidf, x_test_tfidf)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB()
    }
    feature_sets = {
        'BoW': (x_train_bow, x_test_bow),
        'BoW Scaled': (x_train_bow_scaled, x_test_bow_scaled),
        'TF-IDF': (x_train_tfidf, x_test_tfidf),
        'TF-IDF Scaled': (x_train_tfidf_scaled, x_test_tfidf_scaled)
    }
    results = []
    for model_name, model in models.items():
        for feature_name, (x_train_feat, x_test_feat) in feature_sets.items():
            model.fit(x_train_feat, y_train)
            y_pred = model.predict(x_test_feat)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results.append({
                'Model': model_name,
                'Features': feature_name,
                'Accuracy': accuracy,
                'F1 Score': f1
            })

    results_df = pd.DataFrame(results)
    print(results_df)
