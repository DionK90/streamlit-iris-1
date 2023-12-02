# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 00:48:26 2022

@author: dionk
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.write("""
# Iris Prediction App
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe

def user_input_features():
    sl = st.sidebar.slider('Sepal length (cm)', 4.3, 7.9, 5.)
    sw = st.sidebar.slider('Sepal width (cm)', 2., 4.4, 3.)
    pl = st.sidebar.slider('Petal length (cm)', 1., 6.9, 3.)
    pw = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 1.)    
    data = {'0': sl,
            '1': sw,
            '2': pl,
            '3': pw,
            }
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

# Displays the user input features
st.subheader('User Input features')
st.write(df)

# Reads in saved classification model
load_clf = joblib.load('model.joblib')

# Intermediary result
result = load_clf.named_steps["feature_processing"].transform(df)
st.write(result)

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
iris_species = np.array(['setosa','versicolor','virginica'])
st.write(iris_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
