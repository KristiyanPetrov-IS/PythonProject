"""File with APIs that allow experimentation with Word2Vec and GloVe on the data"""
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.defines import DEPENDENT_FIELD, INDEPENDENT_FIELD_TOKEN
from src.defines import W2V_FILE_NAME, INDEPENDENT_FIELD_W2V
from src.defines import GLOVE_FILE_NAME, GLOVE_LEN, INDEPENDENT_FIELD_GLOVE

def document_vector(word2vec_model, doc):
    """Helps with tranformation for Independent field for ML based on W2V model"""
    doc = [word for word in doc if word in word2vec_model.wv]
    if len(doc) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word2vec_model.wv[doc], axis=0)

def eval_vec_model(x, y):
    """Train a randomforestclassifier and evaluate its accuracy"""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)

def experiment_wordtovec(df, train):
    """Main function for w2v functionality"""
    if train:
        word2vec_model = Word2Vec(sentences=df[INDEPENDENT_FIELD_TOKEN],
                                  vector_size=100,
                                  window=5,
                                  min_count=1,
                                  workers=4)
        word2vec_model.save(W2V_FILE_NAME)
    else:
        word2vec_model = Word2Vec.load(W2V_FILE_NAME)

    df[INDEPENDENT_FIELD_W2V] = df[INDEPENDENT_FIELD_TOKEN].apply(
        lambda x: document_vector(word2vec_model, x))

    x = np.array(df[INDEPENDENT_FIELD_W2V].tolist())
    y = df[DEPENDENT_FIELD]

    accuracy = eval_vec_model(x, y)
    print(f"Word2Vec - Accuracy: {accuracy * 100:.2f}%")

    return df

def load_glove_model(glove_file):
    """Function to help load the glove model"""
    with open(glove_file, 'r', encoding='utf-8') as f:
        model = {}
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = [float(val) for val in split_line[1:]]
            model[word] = embedding
    return model

def get_glove_vectors(tokens, glove_model, vector_size=GLOVE_LEN):
    """Helps with tranformation for Independent field for ML based on GloVe model"""
    vectors = []
    for token in tokens:
        if token in glove_model:
            vectors.append(ensure_fixed_length(glove_model[token]))
        else:
            vectors.append(np.zeros(vector_size))

    return vectors

def ensure_fixed_length(vector, fixed_length=GLOVE_LEN):
    """Helps with list lengths to be equal"""
    if len(vector) < fixed_length:
        vector.extend([0] * (fixed_length - len(vector)))
    elif len(vector) > fixed_length:
        vector = vector[:fixed_length]
    return vector

def get_average_vector(vectors):
    """Helps with tranformation for Independent field for ML based on GloVe model"""
    return np.mean(vectors, axis=0)

def experiment_glove(df):
    """Main function for glove functionality"""
    glove_model = load_glove_model(GLOVE_FILE_NAME)
    df[INDEPENDENT_FIELD_GLOVE] = df[INDEPENDENT_FIELD_TOKEN].apply(
        lambda x: get_glove_vectors(x, glove_model))
    df[INDEPENDENT_FIELD_GLOVE] = df[INDEPENDENT_FIELD_GLOVE].apply(get_average_vector)

    x = np.array(df[INDEPENDENT_FIELD_GLOVE].tolist())
    y = df[DEPENDENT_FIELD]

    accuracy = eval_vec_model(x, y)
    print(f"GloVe - Accuracy: {accuracy * 100:.2f}%")

    return df
