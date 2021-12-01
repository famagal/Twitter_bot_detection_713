import requests
import pandas as pd
import numpy as np
import utils

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


    ### time processing
    user_processed['created_at'] = pd.to_datetime(user_processed['created_at'])

    ### renaming columns
    user_processed = user_processed.rename(
        columns={
            'id': 'author_id',
            'name': 'user_display_name',
            'description': 'user_desc',
            'created_at': 'user_created_at',
            'verified': 'user_verified',
            'protected': 'user_private',
            'followers_count': 'user_followers_cnt',
            'following_count': 'user_following_cnt',
            'tweet_count': 'user_tweet_count',
            'listed_count': 'user_list_count'
        })

    ### join in lag_hours (when available)
    if 'empty' not in tweet_df.columns:
        # locally create keep_non_zero function from utils for use
        def keep_non_zero(x):
            if x < 0:
                x = None
            return x
        # join
        tweet_df = tweet_df[['author_id', 'lag']]
        tweet_df['lag_hours'] = (pd.to_numeric((tweet_df['lag'] / 60 / 60)) /
                                 1000000000).apply(utils.keep_non_zero)
        tweet_jn_lag = tweet_df[['author_id', 'lag_hours']].groupby(
            'author_id', as_index=False).agg({'lag_hours': 'std'})
        user_processed = user_processed.merge(tweet_jn_lag,
                                              on='author_id',
                                              how='left')
        # rename column
        user_processed = user_processed.rename(
            columns={'lag_hours': 'lag_hours_std'})
    else:
        user_processed['lag_hours_std'] = np.nan

        # relabel binary columns
    user_processed['user_verified'] = user_processed['user_verified'].map(
        lambda x: 1 if x == True else 0)
    user_processed['user_private'] = user_processed['user_private'].map(
        lambda x: 1 if x == True else 0)

    # keep columns of interest
    user_processed = user_processed[[
        'author_id', 'user_verified', 'user_private', 'user_followers_cnt',
        'user_following_cnt', 'user_tweet_count', 'user_list_count',
        'lag_hours_std'
    ]]

    return user_processed


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
