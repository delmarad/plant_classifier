import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import cv2
import os

def get_TF_version():
    '''Returns the version of the tensorflow library that we use'''
    return tf.__version__

def load_models():
    '''Load the pretrained keras models from local disk to allow predictions. It expects the models under relative subfolder _models_
    
    Returns: a dict of keras models'''

    models = {}
    
    models['Health_lenet'] = keras.models.load_model('models/lenet_Health')
    models['Health_cnn'] = keras.models.load_model('models/cnn_Health')

    models['Plant_lenet'] = keras.models.load_model('models/lenet_Plant')
    models['Plant_custom'] = keras.models.load_model('models/custom_Plant')

    models['PlantDesease_lenet'] = keras.models.load_model('models/lenet_PlantDisease')
    models['PlantDesease_custom'] = keras.models.load_model('models/custom_PlantDisease')
    models['PlantDesease_custom_gen'] = keras.models.load_model('models/custom_gen_PlantDisease')
    models['PlantDesease_resnet'] = keras.models.load_model('models/resnet_PlantDisease')

    return models

def load_validation_metadata(test_image_path):
    '''Walks through the entire validation set on local disk and gathers file metadata
    
    parameters:
    
    test_image_path - the relative base folder where the validation set is stored
    
    Returns a dataframe containing the metadata (the filename acts as unique id thus is used as index)'''

    df = pd.DataFrame(columns=['filepath', 'filename', 'health_label', 'plant_label', 'plant_desease_label'])
    
    for path, dirs, files in os.walk(test_image_path):

        for dir_name in dirs:
            path=test_image_path + dir_name
            
            plant_name = dir_name.split('_')[0]

            if 'healthy' in dir_name.split('___')[1]:
                health_label = 'Healthy'
            else:
                health_label = 'Not Healthy'
            
            for image_file in os.listdir(path):
                
                df.loc[len(df)] = [path  + os.sep + image_file, image_file, health_label, plant_name, dir_name]

    return df.set_index('filename')

def get_dataframe_from_filelist(files, validationset_metadata):
    '''Filters the overall dataset dataframe down to the selected files
    
    parameters:
    
    files  - list of selected files out of the validation set
    validationset_metadata - dataframe containing the entire validation set
    
    Returns a filtered dataframe containing rows only that correspond to the selected files'''
    # extract filenames from file list
    try:
        files = [file.name for file in files]
    except AttributeError:
        pass
    
    # select rows which match the filelist
    return validationset_metadata.loc[files]


def get_predictions(models, data):
    '''Computes the predictions on various models
    
    parameters:
    
    models - dict that holds the pretrained keras models
    data - the dataframe containing the image metadata which acts as prediction input
    
    Returns extended dataframe containing all predected labels and probabilities (confidence) per model and image'''

    # first load selected images from dataframe into memory 
    X_test = []

    width = 80
    height = 80
    
    for index, path in data['filepath'].items():

        img=cv2.imread(path, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (width,height))
        X_test.append(np.array(img_resized))


    X_test = np.array(X_test) # Transfoming the list into arrays
    X_test = X_test.astype('float32') #Setting type float16 to use less memory
    X_test = X_test / 255   # Normalizing the data


    
    # 
    # Health classification using binary classes (Healthy/Not Healthy)
    #

    # binary label re-encoder
    reencode_labels = lambda x: 'Healthy' if x <= 0.5 else 'Not Healthy'
    transform_proba = lambda x: 1-x if x<= 0.5 else x

    # lenet prediction
    data['health_lenet_proba'] = models['Health_lenet'].predict(X_test, verbose=0)
    data['health_lenet_prediction'] = data['health_lenet_proba'].apply(reencode_labels)
    data['health_lenet_proba'] = data['health_lenet_proba'].apply(transform_proba)


    # cnn prediction
    data['health_cnn_proba'] = models['Health_cnn'].predict(X_test, verbose=0)
    data['health_cnn_prediction'] = data['health_cnn_proba'].apply(reencode_labels)
    data['health_cnn_proba'] = data['health_cnn_proba'].apply(transform_proba)

    
    
    #
    # Plant classification using 14 classes
    #

    # plant class labeles re-encoder
    reencode_labels = lambda x: ['Apple', 'Blueberry', 'Cherry','Corn', 'Grape','Orange', 'Peach','Pepper', 'Potato','Raspberry', 'Soybean','Squash','Strawberry','Tomato'][x]

    # lenet
    y_pred = models['Plant_lenet'].predict(X_test, verbose=0)
    data['plant_lenet_prediction'] = np.argmax(y_pred, axis=1)
    data['plant_lenet_prediction'] = data['plant_lenet_prediction'].apply(reencode_labels)
    data['plant_lenet_proba'] = np.amax(y_pred, axis=1)


    # custom    
    y_pred = models['Plant_custom'].predict(X_test, verbose=0)
    data['plant_custom_prediction'] = np.argmax(y_pred, axis=1)
    data['plant_custom_prediction'] = data['plant_custom_prediction'].apply(reencode_labels)
    data['plant_custom_proba'] = np.amax(y_pred, axis=1)
    

    #
    # plant-desease classification with 38 classes (AppleScrab, AppleCedarRust, etc.)
    #

    # plant class-desease labeles re-encoder
    reencode_labels = lambda x: [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'][x]
    
    

    # lenet
    y_pred = models['PlantDesease_lenet'].predict(X_test, verbose=0)
    data['plant_desease_lenet_prediction'] = np.argmax(y_pred, axis=1)
    data['plant_desease_lenet_prediction'] = data['plant_desease_lenet_prediction'].apply(reencode_labels)
    data['plant_desease_lenet_proba'] = np.amax(y_pred, axis=1)

    # custom
    y_pred = models['PlantDesease_custom'].predict(X_test, verbose=0)
    data['plant_desease_custom_prediction'] = np.argmax(y_pred, axis=1)
    data['plant_desease_custom_prediction'] = data['plant_desease_custom_prediction'].apply(reencode_labels)
    data['plant_desease_custom_proba'] = np.amax(y_pred, axis=1)

    # custom_gen
    y_pred = models['PlantDesease_custom_gen'].predict(X_test, verbose=0)
    data['plant_desease_custom_gen_prediction'] = np.argmax(y_pred, axis=1)
    data['plant_desease_custom_gen_prediction'] = data['plant_desease_custom_gen_prediction'].apply(reencode_labels)
    data['plant_desease_custom_gen_proba'] = np.amax(y_pred, axis=1)

    
    
    # resnet

    # reload images and resize (resnet model was trained with 224x224 instead of 80x80 like the other models)
    width = 224
    height = 224
    
    X_test = []

    for index, path in data['filepath'].items():

        img=cv2.imread(path, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (width,height))
        X_test.append(np.array(img_resized))


    X_test = np.array(X_test) # Transfoming the list into arrays
    X_test = X_test.astype('float32') #Setting type float16 to use less memory
    X_test = X_test / 255   # Normalizing the data


    y_pred = models['PlantDesease_resnet'].predict(X_test, verbose=0)
    data['plant_desease_resnet_prediction'] = np.argmax(y_pred, axis=1)
    data['plant_desease_resnet_prediction'] = data['plant_desease_resnet_prediction'].apply(reencode_labels)
    data['plant_desease_resnet_proba'] = np.amax(y_pred, axis=1)

    print("Predictions done for images:", len(data))
    return data


def get_metrics(y_true, y_pred):
    '''Compute classification report and plot confusion matrix
    
    parameters:
    
    y_true - vector containing the true labels
    y_pred - vector containing the predicted labels
    
    returns a tuple consisting of the formatted classificatiob report string and the matplotlib figure object for plotting the confusion matrix'''
    return classification_report(y_true, y_pred, zero_division=0), ConfusionMatrixDisplay.from_predictions(y_true, y_pred, xticks_rotation=45).figure_

def predicted_df_for_export(predicted_df):
    '''Reorder and filter the columns of the prediction dataframe for export'''
    return predicted_df[
            ['health_label', 
            'health_lenet_prediction', 
            'health_lenet_proba', 
            'health_cnn_prediction', 
            'health_cnn_proba', 
            'plant_label', 
            'plant_lenet_prediction',
            'plant_lenet_proba', 
            'plant_custom_prediction',
            'plant_custom_proba', 
            'plant_desease_label', 
            'plant_desease_lenet_prediction', 
            'plant_desease_lenet_proba', 
            'plant_desease_custom_prediction', 
            'plant_desease_custom_proba', 
            'plant_desease_custom_gen_prediction', 
            'plant_desease_custom_gen_proba', 
            'plant_desease_resnet_prediction', 
            'plant_desease_resnet_proba']
        ]