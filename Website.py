import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn
import xgboost

from sklearn.base import BaseEstimator, TransformerMixin
class MyCustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param=1):
        self.param = param

model=pickle.load(open('Model.pkl','rb'))

if "reset" not in st.session_state:
    st.session_state.reset = False

house=pd.read_csv('Cleaned_data.csv')
area_type = house['area_type'].unique()
availability = house['availability'].unique()
location = house['location'].unique()

st.title("House Price Predictor")

area_type_1 = st.selectbox("area type does your house belong to?", area_type)
availability_1 = st.selectbox("Is it availble now or will it be availble soon?", availability)
location_1 = st.selectbox("Which location is the house at?", location)
bath_1 = st.number_input("How many baths does the house have?", value=None, placeholder="Type a number...")
bhk_1 = st.number_input("How many rooms does the house have?", value=None, placeholder="Type a number...")
sqft_1 = st.number_input("What is the sqft of the house?", value=None, placeholder="Type a number...", min_value=300)

if st.button('Predict'):
    prediction=model.predict(pd.DataFrame(columns=["area_type", "availability", "location", "bath", "bhk", "sqft"], 
                                          data=np.array([area_type_1, availability_1, location_1, bath_1, bhk_1, sqft_1]).reshape(1, 6)))
    st.text("The house may cost approximately Rs. "+str(int(prediction[0])*100000))
