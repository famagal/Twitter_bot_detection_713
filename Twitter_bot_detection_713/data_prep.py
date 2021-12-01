import pandas as pd
from Twitter_bot_detection_713.utils import count_mentions, encoding_reply, keep_non_zero
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
from google.cloud import storage


def tweet_df_cleaner(df):

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

    return df


def user_df_cleaner(user_df):

    ### dropping the first useless column
    user_df = user_df.drop(
        columns=['Unnamed: 0', 'profile_image_url', 'location'])

    ### time processing
    user_df['created_at'] = pd.to_datetime(user_df['created_at'])

    ### renaming columns
    user_df = user_df.rename(
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
            'listed_count': 'user_list_count',
            'account_type': 'target'
        })


    ### join in lag_hours
    # join
    tweet_df = pd.read_parquet('gs://tweet-project-713/data/tweets_df.parquet')
    tweet_clean = tweet_df_cleaner(tweet_df)
    tweet_clean = tweet_clean[['author_id', 'lag']]
    tweet_clean['lag_hours'] = (pd.to_numeric(
        (tweet_clean['lag'] / 60 / 60)) / 1000000000).apply(keep_non_zero)
    tweet_jn_lag = tweet_clean[['author_id', 'lag_hours']].groupby(
        'author_id', as_index=False).agg({'lag_hours': 'std'})
    user_df = user_df.merge(tweet_jn_lag, on='author_id', how='left')
    # rename column
    user_df = user_df.rename(columns={'lag_hours': 'lag_hours_std'})


    # relabel binary columns
    user_df['user_verified'] = user_df['user_verified'].map(
        lambda x: 1 if x == True else 0)
    user_df['user_private'] = user_df['user_private'].map(lambda x: 1
                                                          if x == True else 0)
    user_df['target'] = user_df['target'].map(lambda x: 1 if x == 'bot' else 0)


    # keep columns of interest
    user_df = user_df[[
        'author_id', 'user_verified', 'user_private', 'user_followers_cnt',
        'user_following_cnt', 'user_tweet_count', 'user_list_count',
        'lag_hours_std', 'target'
    ]]

    return user_df


def get_final_tweet_data(en=False, write_to_parquet=False):

    tweet_df = pd.read_parquet('../Twitter_bot_detection_713/data/tweets_df.parquet')
    user_df = pd.read_csv('../raw_data/users_data.csv',
                          sep='\t',
                          lineterminator='\n')
    t_df = tweet_df_cleaner(tweet_df)
    u_df = user_df_cleaner(user_df)
    target_join = u_df[['author_id', 'target']]
    t_joined = t_df.merge(target_join, on='author_id', how='left')
    if en:
        t_joined = t_joined[t_joined['lang'] == 'en']
    if write_to_parquet:
        t_joined.to_parquet('../Twitter_bot_detection_713/data/tweets_final.parquet')
        u_df.to_parquet('../Twitter_bot_detection_713/data/users_final.parquet')

    return t_joined

def get_embeded_data(nrows='all', load_from_gcp=False):

    if load_from_gcp:

        storage_client = storage.Client()
        bucket = storage_client.bucket('tweet-project-713')

        blob1 = bucket.blob('data/X_train_embed.pickle')
        pickle_in = blob1.download_as_string()
        X_train_embed = pickle.loads(pickle_in)
        blob2 = bucket.blob('data/X_test_embed.pickle')
        pickle_in = blob2.download_as_string()
        X_test_embed = pickle.loads(pickle_in)
        blob3 = bucket.blob('data/y_train.pickle')
        pickle_in = blob3.download_as_string()
        y_train = pickle.loads(pickle_in)
        blob4 = bucket.blob('data/y_test.pickle')
        pickle_in = blob4.download_as_string()
        y_test = pickle.loads(pickle_in)
        print('data loaded succesfully from gcp')


    else:

        pickle_in1 = open("data/pickled_data/X_train_embed_25.pickle", "rb")
        X_train_embed = pickle.load(pickle_in1)
        pickle_in2 = open("data/pickled_data/X_test_embed_25.pickle", "rb")
        X_test_embed = pickle.load(pickle_in2)
        pickle_in3 = open("data/pickled_data/y_train.pickle", "rb")
        y_train = pickle.load(pickle_in3)
        pickle_in4 = open("data/pickled_data/y_test.pickle", "rb")
        y_test = pickle.load(pickle_in4)



    if nrows == 'all':

        X_test_pad = pad_sequences(X_test_embed,
                               dtype='float32',
                               padding='post',
                               maxlen=60)
        X_train_pad = pad_sequences(X_train_embed,
                                dtype='float32',
                                padding='post',
                                maxlen=60)
        return X_train_pad, X_test_pad, y_train, y_test

    if type(nrows) == int:
        print(X_test_embed.shape)
        X_test_pad = pad_sequences(X_test_embed[0:round(nrows*0.2)],
                                   dtype='float32',
                                   padding='post',
                                   maxlen=60)
        X_train_pad = pad_sequences(X_train_embed[0:nrows],
                                    dtype='float32',
                                    padding='post',
                                    maxlen=60)

        return X_train_pad, X_test_pad, y_train[0:nrows], y_test[0:round(nrows*0.2)]


def get_user_training_data(load_from_gcp=False):
    # Read and clean data

    if load_from_gcp:

        df = pd.read_csv('gs://tweet-project-713/data/users_data.csv',
                         sep='\t',
                         lineterminator='\n')

    else:
        df = pd.read_csv('../raw_data/users_data.csv',
                         sep='\t',
                         lineterminator='\n')


    df = user_df_cleaner(df)

    # Balancing data
    df_t1 = df[df['target'] == 1]
    df_t0 = df[df['target'] == 0]
    df_t0 = df_t0.sample(len(df_t1), random_state=7)
    df = df_t0.append(df_t1)

    # Train/test split
    X = df.drop(columns='target')
    y = df['target']
    X_train_user, x_test_user, y_train_user, y_test_user = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)

    return X_train_user, x_test_user, y_train_user, y_test_user
