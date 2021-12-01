from Twitter_bot_detection_713.new_preprocessing_text import apply_text_cleaning
from Twitter_bot_detection_713.data_prep import get_embeded_data, get_user_training_data
from Twitter_bot_detection_713.DL_architectures import initialize_model_rnn2_25, initialize_model_rnn_big
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from Twitter_bot_detection_713.gcp_training import save_model_to_gcp, save_model, save_nn, save_nn_to_gcp

class Trainer():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.pipeline = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def set_pipeline(self):

        # for num columns needing both imputation AND scaling
        num_imputer_scaler = Pipeline([
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())])

        # for num columns needing ONLY scaling
        num_scaler = Pipeline([('scaler', RobustScaler())])

        # Applying transformer into the preprocessor
        preprocessor = ColumnTransformer([
            ('num_imputer_scaler', num_imputer_scaler, ['lag_hours_std']),
            ('num_scaler', num_scaler, ['user_followers_cnt',
                                        'user_following_cnt',
                                        'user_tweet_count',
                                        'user_list_count'])
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
    X_train_pad, X_test_pad, y_train, y_test = get_embeded_data(nrows='all', load_from_gcp=True)
    es = EarlyStopping(patience=5, restore_best_weights=True)
    print('initializing model...')
    text_model = initialize_model_rnn_big()
    print('training_model...')
    history = text_model.fit(X_train_pad,
                                    y_train,
                                    epochs=15,
                                    batch_size=128,
                                    validation_split=0.3,
                                    callbacks=[es])

    save_nn_to_gcp(text_model, 'RNN_BIG')
    print('USER MODEL RUNNING')
    print('loading user training data...')
    X_train_user, X_test_user, y_train_user, y_test_user = get_user_training_data(load_from_gcp=True
    )
    print('training model')
    user_trainer = Trainer(X_train=X_train_user, y_train=y_train_user, X_test=X_test_user, y_test=y_test_user)
    user_trainer.run()
    save_model_to_gcp(user_trainer, 'Logit_opt')
    print('EVALUATING TEXT MODEL...')
    print(text_model.evaluate(X_test_pad,y_test))
    print('EVALUATING USER MODEL...')
    user_trainer.evaluate()
