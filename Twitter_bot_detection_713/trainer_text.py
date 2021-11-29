from Twitter_bot_detection_713.get_tweet_data import get_tweet_data
from Twitter_bot_detection_713.new_preprocessing_text import apply_text_cleaning
from Twitter_bot_detection_713.data_prep import get_embeded_data
from Twitter_bot_detection_713.DL_architectures import initialize_model_rnn2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression


class Trainer():

    def __init__(self, X_train, y_train, X_test, y_test):
        self.pipe = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def set_pipeline(self):
       preprocessor = ColumnTransformer([('num_tr', RobustScaler(), ['user_followers_cnt',
                                        'user_following_cnt', 'user_tweet_count', 'user_list_count'])])

        pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('log_reg', LogisticRegression())
                ])

        df_transformed = pipe.fit(X_train, y_train)

        return df_transformed




    if __name__ == "__main__":
        print('loading embedded data...')
        X_train_pad, X_test_pad, y_train, y_test = get_embeded_data(nrows=100)
        es = EarlyStopping(patience=5, restore_best_weights=True)
        print('initializing model...')
        model = initialize_model_rnn2()
        print('training_model...')
        history = model.fit(X_train_pad,
                                    y_train,
                                    epochs=30,
                                    batch_size=16,
                                    validation_split=0.3,
                                    callbacks=[es])
        print('evaluating model...')
        print(model.evaluate(X_test_pad,y_test))
