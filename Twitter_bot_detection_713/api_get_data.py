import requests
import pandas as pd
import numpy as np

headers = ''


def user_data_request(username):
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

def tweet_data_request(user_processed):
    user_processed['id']
    try:
        request.get('')['data']
    except:
        return pd.DataFrame({'empty':0})


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
