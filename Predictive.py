# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/deepa/OneDrive/Machine Learning ( ML )/- ML Projects/2. DIabetes Prediction/trained_model.sav', 'rb'))

# Making a PREDICTIVE Data -

input_data = (7,147,76,0,0,39.4,0.257,43)

# Changing the input data to numpy array -
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance -
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

# Standardize the input data -
std_data = scaler.transform(input_data_reshape)
print(std_data,'\n')

prediction = loaded_model.predict(input_data_reshape)
print(prediction, '\n')

if (prediction[0] == 0) :
    print('The person is non-diabetic')
else:
    print('The person is Diabetic')