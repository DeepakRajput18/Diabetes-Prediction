# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 22:42:28 2025

@author: deepa
"""

import numpy as np
import pickle
import streamlit as st

# Loading the Saved File -
loaded_model = pickle.load(open('C:/Users/deepa/OneDrive/Machine Learning ( ML )/- ML Projects/2. DIabetes Prediction/trained_model.sav', 'rb'))

# Creating a function for Prediction -

def diabetes_Prediction(input_data) :
    
    input_data = (7,147,76,0,0,39.4,0.257,43)

    # Changing the input data to numpy array -
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance -
    input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshape)
    print(prediction, '\n')

    if (prediction[0] == 0) :
        return 'The person is non-diabetic'
    else:
        return'The person is Diabetic'
        

def main():
    
    # Giving a title
    st.title('Diabetes Prediction Web App')
    
    # Getting the input Data from Users -
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input(' BMI Valuel')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of Person')
    
    
    # Code for Prediction -
    diagnosis = ''
    
    # Creating a button for Prediction -
    if st.button('Diabetes Test Result :'):
        
        diagnosis = diabetes_Prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        st.success(diagnosis)
        
        
if __name__ == '__main__' :
    main()

