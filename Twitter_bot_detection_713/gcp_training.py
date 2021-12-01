import joblib
import pandas as pd
from google.cloud import storage
from tensorflow.keras.models import save_model


BUCKET_NAME = 'tweet-project-713'

BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'

MODEL_NAME = 'twitter_bot_detector'

MODEL_VERSION = 'v1'

def save_nn_to_gcp(model, model_name):
    model.save(f'{model_name}.h5')
    local_model_name = f'{model_name}.h5'
    print("saved NN locally")
    client = storage.Client().bucket(BUCKET_NAME)
    storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
    blob = client.blob(storage_location)
    blob.upload_from_filename(local_model_name)
    print("uploaded model.joblib to gcp cloud storage under \n => {}".format(
        storage_location))


def save_model_to_gcp(model, model_name):
    """Save the model into a .joblib and upload it on Google Storage /models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""
    from sklearn.externals import joblib
    local_model_name = f'{model_name}.joblib'
    # saving the trained model to disk (which does not really make sense
    # if we are running this code on GCP, because then this file cannot be accessed once the code finished its execution)
    joblib.dump(model, local_model_name)
    print("saved model.joblib locally")
    client = storage.Client().bucket(BUCKET_NAME)
    storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
    blob = client.blob(storage_location)
    blob.upload_from_filename(local_model_name)
    print("uploaded model.joblib to gcp cloud storage under \n => {}".format(
        storage_location))



def save_nn(fitted_model, model_name):
    fitted_model.save(f'{model_name}.h5')
    print(f"saved {model_name}.h5 locally")


def save_model(fitted_model, model_name):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(fitted_model, f'{model_name}.joblib')
    print(f"saved {model_name}.joblib locally")
