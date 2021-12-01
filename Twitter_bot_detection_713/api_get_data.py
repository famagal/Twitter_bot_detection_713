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

def tweet_data_request(user_id):
    pass
