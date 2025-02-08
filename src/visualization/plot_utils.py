"""Module provides APIs to visualize plots for data"""
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from src.defines import DEPENDENT_FIELD, INDEPENDENT_FIELD_LEN
from src.defines import INDEPENDENT_FIELD_LEMMAT, INDEPENDENT_FIELD_CLEAN
from src.defines import INDEPENDENT_FIELD_TOKEN, INDEPENDENT_FIELD_SENTIMENT

def plot_rating_counts(df):
    """Rating countplot"""
    sns.countplot(x=DEPENDENT_FIELD, data=df)
    plt.title('Разпределение на рейтингите')
    plt.show()

def plot_review_length_distribution(df):
    """Rating and review length boxplot"""
    sns.boxplot(x=DEPENDENT_FIELD, y=INDEPENDENT_FIELD_LEN, data=df)
    plt.title('Дължина на ревютата спрямо рейтинга')
    plt.show()

def plot_most_frequent_words(df, has_lemmatization = False):
    """Barplot of 20 most frequent words in reviews"""
    if has_lemmatization:
        all_words = ' '.join(df[INDEPENDENT_FIELD_LEMMAT]).split()
    else:
        all_words = ' '.join(df[INDEPENDENT_FIELD_CLEAN]).split()
    word_freq = Counter(all_words)
    common_words = word_freq.most_common(20)
    sns.barplot(x=[word[0] for word in common_words], y=[word[1] for word in common_words])
    plt.title('Честота на най-често срещаните думи')
    plt.xticks(rotation=45)
    plt.show()

def plot_corr_rating_reviewlen(df):
    """Heatmap for review length and rating"""
    sns.heatmap(df[[INDEPENDENT_FIELD_LEN, DEPENDENT_FIELD]].corr(), annot=True, cmap='coolwarm')
    plt.title('Корелация между дължината на ревюто и рейтинга')
    plt.show()

def plot_review_len_hist(df):
    """Histplot of review length in 20 bins"""
    sns.histplot(df[INDEPENDENT_FIELD_LEN], bins=20)
    plt.title('Разпределение на думите в ревютата')
    plt.show()

def plot_rating_to_reviewlen_ratio(df):
    """Violin plot of review length and rating"""
    sns.violinplot(x=DEPENDENT_FIELD, y=INDEPENDENT_FIELD_LEN, data=df)
    plt.title('Разпределение на рейтингите спрямо дължината на ревюто')
    plt.show()

def plot_most_frequent_positive_words(df):
    """Plot for most frequent words in positive reviews"""
    positive_words = [word for tokens in
                      df[df[DEPENDENT_FIELD] >= 4][INDEPENDENT_FIELD_TOKEN] for word in tokens]
    positive_freq = nltk.FreqDist(positive_words)
    positive_freq.plot(10, cumulative=False, title='Топ 10 думи за положителни ревюта')
    plt.show()

def plot_most_frequent_negative_words(df):
    """Plot for most frequent words in negative reviews"""
    negative_words = [word for tokens in
                      df[df[DEPENDENT_FIELD] <= 2][INDEPENDENT_FIELD_TOKEN] for word in tokens]
    negative_freq = nltk.FreqDist(negative_words)
    negative_freq.plot(10, cumulative=False, title='Топ 10 думи за отрицателни ревюта')
    plt.show()

def plot_most_frequent_words_per_rating(df):
    """Barplot per rating for most frequent words in positive reviews in the rating"""
    for rating in df[DEPENDENT_FIELD].unique():
        words = [word for tokens in
                 df[df[DEPENDENT_FIELD] == rating][INDEPENDENT_FIELD_TOKEN] for word in tokens]
        word_freq = nltk.FreqDist(words)

        sns.barplot(x=[word[0] for word in word_freq.most_common(10)],
                    y=[word[1] for word in word_freq.most_common(10)])
        plt.title(f'Топ 10 думи за рейтинг {rating}')
        plt.show()

def plot_pos_bigrams(df):
    """Plot for most common bigrams in positive reviews"""
    positive_words = [word for tokens in
                      df[df[DEPENDENT_FIELD] >= 4][INDEPENDENT_FIELD_TOKEN] for word in tokens]
    bigrams = list(nltk.bigrams(positive_words))
    bigram_freq = nltk.FreqDist(bigrams)

    plt.figure(figsize=(12, 4))
    bigram_freq.plot(10, cumulative=False)
    plt.title('Топ 10 положителни биграми')
    plt.show()

def plot_neg_bigrams(df):
    """Plot for most common bigrams in negative reviews"""
    negative_words = [word for tokens in
                      df[df[DEPENDENT_FIELD] <= 2][INDEPENDENT_FIELD_TOKEN] for word in tokens]
    bigrams = list(nltk.bigrams(negative_words))
    bigram_freq = nltk.FreqDist(bigrams)

    plt.figure(figsize=(12, 4))
    bigram_freq.plot(10, cumulative=False)
    plt.title('Топ 10 негативни биграми')
    plt.show()

def plot_pos_trigrams(df):
    """Plot for most common trigrams in positive reviews"""
    positive_words = [word for tokens in
                      df[df[DEPENDENT_FIELD] >= 4][INDEPENDENT_FIELD_TOKEN] for word in tokens]
    trigrams = list(nltk.trigrams(positive_words))
    trigram_freq = nltk.FreqDist(trigrams)

    plt.figure(figsize=(12, 4))
    trigram_freq.plot(10, cumulative=False)
    plt.title('Топ 10 положителни триграми')
    plt.show()


def plot_neg_trigrams(df):
    """Plot for most common trigrams in negative reviews"""
    negative_words = [word for tokens in
                      df[df[DEPENDENT_FIELD] <= 2][INDEPENDENT_FIELD_TOKEN] for word in tokens]
    trigrams = list(nltk.trigrams(negative_words))
    trigram_freq = nltk.FreqDist(trigrams)

    plt.figure(figsize=(12, 4))
    trigram_freq.plot(10, cumulative=False)
    plt.title('Топ 10 негативни триграми')
    plt.show()

def plot_sentiment_dist(df):
    """Histplot for sentiment in 20 bins"""
    sns.histplot(df[INDEPENDENT_FIELD_SENTIMENT], bins=20, color='blue')
    plt.title('Разпределение на Настроения')
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.show()

def plot_sentiment_to_rating(df):
    """Boxplot and violinplot for sentiment and rating"""
    sns.boxplot(x=DEPENDENT_FIELD, y=INDEPENDENT_FIELD_SENTIMENT, data=df)
    plt.title('Настроения спрямо оценка')
    plt.show()

    sns.violinplot(x=DEPENDENT_FIELD, y=INDEPENDENT_FIELD_SENTIMENT, data=df)
    plt.title('Разпределение на ревю спрямо оценка и настроение')
    plt.show()

def visualize_plots(df, has_lemmatization = False):
    """Execute all plots on the dataframe"""
    plot_rating_counts(df)
    plot_review_length_distribution(df)
    plot_most_frequent_words(df, has_lemmatization)
    plot_corr_rating_reviewlen(df)
    plot_review_len_hist(df)
    plot_rating_to_reviewlen_ratio(df)
    plot_most_frequent_positive_words(df)
    plot_most_frequent_negative_words(df)
    plot_most_frequent_words_per_rating(df)
    plot_pos_bigrams(df)
    plot_neg_bigrams(df)
    plot_pos_trigrams(df)
    plot_neg_trigrams(df)
    plot_sentiment_dist(df)
    plot_sentiment_to_rating(df)
