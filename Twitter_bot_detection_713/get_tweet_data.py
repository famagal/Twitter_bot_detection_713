import pandas as pd
import requests
import time
import os
'''Function for getting data from API. Function not used in Trainer.py. All data stored in raw data'''

def get_tweet_data(user_df, headers):
    list_users = list(user_df['id'])
    bins = list(range(0, len(list_users), 900))
    bins.remove(0)

    for index, user in enumerate(list_users):
        if index in bins:
            print('Sleeping...Wait 15 min')
            time.sleep(60 * 15)

        if index == 0:
            try:
                url = f'https://api.twitter.com/2/users/{user}/tweets?exclude=retweets&expansions=attachments.media_keys,author_id,entities.mentions.username,in_reply_to_user_id,referenced_tweets.id&tweet.fields=attachments,created_at,in_reply_to_user_id,lang,possibly_sensitive,public_metrics,referenced_tweets,text&max_results=100'
                response = requests.get(url, headers=headers).json()['data']
                df = pd.DataFrame(response)
            except:
                continue
        else:
            try:
                print(f'Getting data for user {index+1}')
                url = f'https://api.twitter.com/2/users/{user}/tweets?exclude=retweets&expansions=attachments.media_keys,author_id,entities.mentions.username,in_reply_to_user_id,referenced_tweets.id&tweet.fields=attachments,created_at,in_reply_to_user_id,lang,possibly_sensitive,public_metrics,referenced_tweets,text&max_results=100'
                response = requests.get(url, headers=headers).json()['data']
                df = df.append(pd.DataFrame(response))
                df.to_parquet('data/tweets_df.parquet')
                print(f'Added user {index+1}')

            except:
                continue

    print('writing successfull!')


df = pd.read_csv('raw_data/users_data.csv', sep='\t', lineterminator='\n')

headers = {
    'Authorization':
    f"Bearer {os.getenv('BEARER_TOKEN')}"}


###Uncomment if you want to run from the terminal.

###get_tweet_data(df, headers)
