import requests
import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from gensim.models import Word2Vec
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from Twitter_bot_detection_713.utils import *

headers = ''

word2vec = api.load('glove-twitter-200')


def user_data_request(username):

    try:
        response = requests.get(
            f'https://api.twitter.com/2/users/by/username/{username}?user.fields=id,name,username,created_at,description,profile_image_url,location,verified,protected,public_metrics',
            headers=headers).json()
        user = pd.DataFrame(pd.Series(response['data'])).T
        u_org = user[['id', 'username', 'name', 'description', 'created_at', 'verified','protected','public_metrics']]
        u_org['followers_count'] = [i['followers_count'] for i in u_org['public_metrics']]

        u_org['following_count'] = [i['following_count'] for i in u_org['public_metrics']]

        u_org['tweet_count'] = [i['tweet_count'] for i in u_org['public_metrics']]

        u_org['listed_count'] = [i['listed_count'] for i in u_org['public_metrics']]

        user_processed = u_org[[
            'id', 'username', 'name', 'description', 'created_at',
            'verified', 'protected', 'followers_count',
            'following_count', 'tweet_count', 'listed_count'
        ]]
        return user_processed
    except:
        return 'No user data available. Check username for misspellings'

def tweet_data_request(user_processed):
    user = user_processed['id'][0]
    try:
        url = f'https://api.twitter.com/2/users/{user}/tweets?exclude=retweets&expansions=attachments.media_keys,author_id,entities.mentions.username,in_reply_to_user_id,referenced_tweets.id&tweet.fields=attachments,created_at,in_reply_to_user_id,lang,possibly_sensitive,public_metrics,referenced_tweets,text&max_results=100'
        response = requests.get(url, headers=headers).json()['data']
        template = pd.read_parquet('data/template_twdf.parquet')
        df = template.append(response)
        df = df[df['author_id'] == user]

        ### time processing

        df = df.sort_values(by=['author_id', 'created_at'],
                            ascending=True,
                            ignore_index=True)

        df['created_at'] = pd.to_datetime(df['created_at'])

        df['lag'] = df.groupby('author_id', as_index=False)['created_at'].diff(
        )  ### this line creates the lag column - difference of time between tweets made by user

        ##contain attachments?

        df['attachments'] = df['attachments'] == df['attachments']

        ##public metrics unpacking

        df['like_count'] = [i['like_count'] for i in df['public_metrics']]

        df['quote_count'] = [i['quote_count'] for i in df['public_metrics']]

        df['reply_count'] = [i['reply_count'] for i in df['public_metrics']]

        df['retweet_count'] = [i['retweet_count'] for i in df['public_metrics']]

        ###entities unpacking

        df['n_mentions'] = df['entities'].apply(count_mentions)

        ##reply category

        df['reply_category'] = df.apply(lambda row: encoding_reply(row), axis=1)

        ##contain references?

        df['referenced_tweets'] = df['referenced_tweets'] == df[
            'referenced_tweets']

        ##converting author id to integer

        df['author_id'] = df['author_id'].astype(int)

        ##organizing df

        df = df[[
            'author_id', 'id', 'lang', 'text', 'created_at', 'lag',
            'possibly_sensitive', 'referenced_tweets', 'reply_category',
            'like_count', 'quote_count', 'reply_count', 'retweet_count',
            'n_mentions'
        ]]

    except:
        print('Could not fetch tweets from user.')
        df = pd.DataFrame(columns={'empty': 0})

    return df

def user_preprocessing(tweet_df, user_processed):
    #if tweet_df['empty'] in tweet_df.keys():
    pass


def tweet_preprocessing(tweet_df):
    pass
##returns padded X_train, padded y_train



### user inputs username
###user_data_request is called
###user_data_request(username)
###returns user_df
###calls tweet_data_request(user_df)
###returns_tweets_df_cleaned = tweets_df
##user_processing(tweets_df, user_processed)
