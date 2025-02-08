"""File containing string defines used in the project"""
SEPARATOR_ONE = "==================================="
SEPARATOR_TWO = "-----------------------------------"

INDEPENDENT_FIELD = 'Review'
DEPENDENT_FIELD = 'Rating'
INDEPENDENT_FIELD_LEN = 'Review_length'
INDEPENDENT_FIELD_CLEAN = 'Cleaned_review'
INDEPENDENT_FIELD_LEMMAT = 'Lemmatized_review'
INDEPENDENT_FIELD_STEM = 'Stemmed_review'
INDEPENDENT_FIELD_TOKEN = 'Tokens'
INDEPENDENT_FIELD_W2V = 'W2V'
INDEPENDENT_FIELD_GLOVE = 'GloVe'
INDEPENDENT_FIELD_SENTIMENT = 'Review_sentiment'

W2V_FILE_NAME = "word2vec.model"
GLOVE_FILE_NAME = "glove.model.txt"

DATA_PATH = 'src/data/tripadvisor_hotel_reviews.csv'

PRETRAINED_BERT = 'bert-base-uncased'
BERT_RETURN_TENSOR ='tf'
BERT_OPTIMIZER = 'adam'
BERT_INPUT_IDS = 'input_ids'
BERT_ATTENTION_MASK = 'attention_mask'

BERT_SENTIMENT_METRIC = 'score'

GLOVE_LEN = 50
