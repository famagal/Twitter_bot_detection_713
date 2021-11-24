import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re


# Function to remove punctuation from a column
def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


# Function to remove punctuation from a column
def lower_case(text):
    return text.lower()


# Function to remove numbers from a column
def remove_numbers(text):
    return ''.join(word for word in text if not word.isdigit())


# Function to remove numbers stopwords from a column
stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    return [w for w in word_tokens if not w in stop_words]


# Functions to remove numbers lemmatize a column, then reconvert from list to string
lemmatizer = WordNetLemmatizer()


def lemmatize(text):
    return [lemmatizer.lemmatize(word) for word in text]


def list_to_string(L):
    return " ".join(str(x) for x in L)


def remove_at_mentions(text):
    return re.sub(r"(@\w+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "",
                  text)

def apply_text_cleaning(df):
# Apply all cleaning transformations
    df['clean_text'] = df['text'].apply(remove_at_mentions).apply(
    remove_punctuation).apply(lower_case).apply(remove_numbers).apply(
        remove_stopwords).apply(lemmatize).apply(list_to_string)
    return df
