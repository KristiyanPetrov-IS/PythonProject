"""Main module"""
import sys
from src.dataprocessing.data_processor import load_data, prepare_df
from src.dataprocessing.text_cleaner import add_processed_fields
from src.visualization.plot_utils import visualize_plots
from src.models.train_models import train_models_and_compare
from src.dataprocessing.vector_models import experiment_wordtovec, experiment_glove
from src.models.bert_model import build_bert_model, bert_v2
from src.defines import DATA_PATH, INDEPENDENT_FIELD_CLEAN
from src.defines import INDEPENDENT_FIELD_LEMMAT, DEPENDENT_FIELD


def main(has_lemmatization, train_w2v, build_bert):
    """ Main function - prepares data and trains ML models on it
    Experiments with vector Models Word2Vec and GloVe
    Plots for data relations
    """
    df = load_data(DATA_PATH)
    df = prepare_df(df)
    df = add_processed_fields(df, has_lemmatization)
    if has_lemmatization:
        train_models_and_compare(x = df[INDEPENDENT_FIELD_LEMMAT],y = df[DEPENDENT_FIELD])
    else:
        train_models_and_compare(x = df[INDEPENDENT_FIELD_CLEAN],y = df[DEPENDENT_FIELD])

    df = experiment_wordtovec(df, train_w2v)
    df = experiment_glove(df)

    if build_bert:
        build_bert_model(df)
        bert_v2(df)

    visualize_plots(df, has_lemmatization)

if __name__ == "__main__":
    LEMM = False
    BERT = False
    W2V = False

    if "--lemmatization" in sys.argv:
        LEMM = True
    if "--train_w2v" in sys.argv:
        BERT = True
    if "--build_bert" in sys.argv:
        W2V = True

    main(LEMM, BERT, W2V)
