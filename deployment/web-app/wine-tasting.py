# -*- coding: utf-8 -*-

# importing useful libraries
import numpy as np
import tensorflow
import random as python_random

# setting random seed
np.random.seed(1)
python_random.seed(12)
tensorflow.random.set_seed(123)

# importing other useful libraries
from flask import Flask, render_template, request
import gzip
import dill
from tensorflow.keras.layers import Embedding, Dense, GlobalMaxPool1D, Conv1D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# avioding dill classtype error
dill._dill._reverse_typemap['ClassType'] = type


wp = Flask(__name__) # initializes a web app

# defining labels dict for decoding predictions
labels_dict = {0: 'Bordeaux-style Red Blend', 1: 'Cabernet Sauvignon', 2: 'Chardonnay', 3: 'Malbec',
 4: 'Merlot', 5: 'Nebbiolo', 6: 'Pinot Gris', 7: 'Pinot Noir', 8: 'Portuguese Red', 9: 'Red Blend',
 10: 'Rhône-style Red Blend', 11: 'Riesling', 12: 'Rosé', 13: 'Sangiovese', 14: 'Sauvignon Blanc',
 15: 'Sparkling Blend', 16: 'Syrah', 17: 'Tempranillo', 18: 'White Blend', 19: 'Zinfandel'}

# defining a function to rebuild our CNN model
def build_cnn_model(embedding_matrix, input_length):
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], 
                           weights=[embedding_matrix], 
                           input_length=input_length,
                           mask_zero=True,
                           trainable=False))
    
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    
    model.add(GlobalMaxPool1D())
    
    model.add(Dropout(0.2))
    
    model.add(Dense(20, activation='softmax'))

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

    return model

# routing home page
@wp.route('/')
def home():
    return render_template('home.html')

# routing varieties page
@wp.route('/varieties')
def varieties():
    return render_template('varieties.html')

# routing prediction page
@wp.route('/predict', methods=["POST"])
def get_prediction():
    inp = request.form["inp"] # takes the prediction from the form in the html
    
    input_length = 87 # setting the input length of the embedding layer of our model
    
    # opening our data preprocessing object
    with gzip.open("wine-tasting-data-prep.dill.gz", "rb") as prep:
        d_prep = dill.load(prep)
    
    # loading our embedding matrix
    with gzip.open("embedding-matrix.dill.gz", "rb") as emb:
        emb_m = dill.load(emb)
    
    # build the CNN model
    cnn = build_cnn_model(emb_m, input_length)
    cnn.load_weights('cnn-model.hdf5') # loads in the weights
    
    X_prep = d_prep.transform([inp]) # preprocess user's input
    
    # validates if user's input is descriptive enough
    try:
        pred = cnn.predict(X_prep)
        
    except ValueError:
        return "Your input is not descriptive enough. Please be more descriptive."
        
    # taking the top 5 predictions
    top_5_pred = np.argsort(pred[0])[-1:-6:-1]
    
    # making results as string
    result1 = "The wine's variety is most likely {} with a probability of {}%.\n".format(\
                        labels_dict[top_5_pred[0]], round(pred[0][top_5_pred[0]]*100, 3))
        
    result2 = '\nOther possible varieties are:\n'

    for i in range(1, len(top_5_pred)):
        result2 += '{} ==> {}%\n'.format(\
                                labels_dict[top_5_pred[i]], round(pred[0][top_5_pred[i]]*100, 3))
    
    # return result.
    result = result1 + result2
    return result1 + result2

# running the web app
if __name__ == '__main__':
    wp.run(debug=True)