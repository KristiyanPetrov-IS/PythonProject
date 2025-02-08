"""File that builds and evaluates a BERT model"""
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import pipeline
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.defines import INDEPENDENT_FIELD, DEPENDENT_FIELD, INDEPENDENT_FIELD_CLEAN
from src.defines import BERT_RETURN_TENSOR, BERT_OPTIMIZER, BERT_INPUT_IDS
from src.defines import BERT_ATTENTION_MASK, BERT_SENTIMENT_METRIC, PRETRAINED_BERT

def build_bert_model(df):
    """Main bert model training, uses CPU and takes a long time"""
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT)
    model = TFBertForSequenceClassification.from_pretrained(PRETRAINED_BERT, num_labels=5)

    x_train, x_test, y_train, y_test = train_test_split(df[INDEPENDENT_FIELD_CLEAN].tolist(),
                                                        df[DEPENDENT_FIELD] - 1,
                                                        test_size=0.2,
                                                        random_state=42)

    train_inputs = tokenizer(x_train, padding=True, truncation=True,
                             return_tensors=BERT_RETURN_TENSOR)
    val_inputs = tokenizer(x_test, padding=True, truncation=True,
                           return_tensors=BERT_RETURN_TENSOR)

    train_labels = tf.keras.utils.to_categorical(y_train, num_classes=5)
    val_labels = tf.keras.utils.to_categorical(y_test, num_classes=5)

    model.compile(optimizer=BERT_OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit({BERT_INPUT_IDS: train_inputs[BERT_INPUT_IDS],
               BERT_ATTENTION_MASK: train_inputs[BERT_ATTENTION_MASK]},
               train_labels,
               validation_data=(
                   {BERT_INPUT_IDS: val_inputs[BERT_INPUT_IDS],
                    BERT_ATTENTION_MASK: val_inputs[BERT_ATTENTION_MASK]},
                   val_labels),
                epochs=3,
                batch_size=16)

    val_loss, val_accuracy = model.evaluate(
        {BERT_INPUT_IDS: val_inputs[BERT_INPUT_IDS],
         BERT_ATTENTION_MASK: val_inputs[BERT_ATTENTION_MASK]},
        val_labels,
        batch_size=16)

    print(f"BERT - Validation Loss: {val_loss * 100:.2f}%")
    print(f"BERT - Validation Accuracy: {val_accuracy * 100:.2f}%")

def sentiment_to_rating(sentiment, inverse):
    """Helper function for score to rating translation in bert_v2"""
    score = sentiment[0][BERT_SENTIMENT_METRIC]

    if score < 0.485:
        return 1 if inverse else 5
    if score < 0.495:
        return 2 if inverse else 4
    if score < 0.505:
        return 3
    if score < 0.515:
        return 4 if inverse else 2
    return 5 if inverse else 1

def bert_v2(df):
    """Second BERT evaluation using sentiment-analysis in pipeline"""
    sentiment_analyzer = pipeline('sentiment-analysis',
                                  model=PRETRAINED_BERT,
                                  tokenizer=PRETRAINED_BERT)

    results = []
    sentiment_scores = []
    for review in df[INDEPENDENT_FIELD]:
        sentiment = sentiment_analyzer(review)
        sentiment_results = sentiment[0]
        sentiment_scores.append(sentiment_results[BERT_SENTIMENT_METRIC])
        results.append(sentiment)

    predicted_ratings = [sentiment_to_rating(sentiment, 0) for sentiment in results]
    predicted_ratings_inversed = [sentiment_to_rating(sentiment, 1) for sentiment in results]

    true_ratings = df[DEPENDENT_FIELD].tolist()

    accuracy = accuracy_score(true_ratings, predicted_ratings)
    print(f"Bert_v2 - Accuracy: {accuracy * 100:.2f}%")

    accuracy = accuracy_score(true_ratings, predicted_ratings_inversed)
    print(f"Bert_v2 - Accuracy_inversed: {accuracy * 100:.2f}%")
