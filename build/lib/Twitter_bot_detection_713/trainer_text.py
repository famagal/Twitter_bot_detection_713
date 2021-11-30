from Twitter_bot_detection_713.new_preprocessing_text import apply_text_cleaning
from Twitter_bot_detection_713.data_prep import get_embeded_data, get_user_training_data
from Twitter_bot_detection_713.DL_architectures import initialize_model_rnn2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class Trainer():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.pipeline = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def set_pipeline(self):

        preprocessor = Pipeline([
            ('num_imputer', SimpleImputer(strategy='median'), ['lag_hours_std']),
            ('num_tr', RobustScaler(), ['user_followers_cnt',
                                        'user_following_cnt',
                                        'user_tweet_count',
                                        'user_list_count',
                                        'lag_hours_std']),
        ])

        self.pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('log_reg', LogisticRegression())
                ])

    def run(self):
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        """evaluates the pipeline on df_test and return the scores"""
        y_pred = self.pipeline.predict(self.X_test)
        print(f'Accuracy user_model 1: {accuracy_score(self.y_test, y_pred)}')
        print(f'Precision user_model 1: {precision_score(self.y_test, y_pred)}')
        print(f'Recall user_model 1: {recall_score(self.y_test, y_pred)}')
        print(f'f1 user_model 1: {f1_score(self.y_test, y_pred)}')




if __name__ == "__main__":
    print('TEXT MODEL RUNNING')
    print('loading embedded data...')
    X_train_pad, X_test_pad, y_train, y_test = get_embeded_data(nrows=100)
    es = EarlyStopping(patience=5, restore_best_weights=True)
    print('initializing model...')
    text_model = initialize_model_rnn2()
    print('training_model...')
    history = text_model.fit(X_train_pad,
                                    y_train,
                                    epochs=30,
                                    batch_size=16,
                                    validation_split=0.3,
                                    callbacks=[es])
    print('USER MODEL RUNNING')
    print('loading user training data...')
    X_train_user, X_test_user, y_train_user, y_test_user = get_user_training_data(
    )
    print('training model')
    user_trainer = Trainer(X_train=X_train_user, y_train=y_train_user, X_test=X_test_user, y_test=y_test_user)
    user_trainer.run()
    print('EVALUATING TEXT MODEL...')
    print(text_model.evaluate(X_test_pad,y_test))
    print('EVALUATING USER MODEL...')
    user_trainer.evaluate()
