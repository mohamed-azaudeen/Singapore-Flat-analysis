import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

url = 'http://192.168.0.169:8000/final_dataset.csv'
df = pd.read_csv(url)
df = df.drop(['Unnamed: 0'],axis=1)


st.title(":red[Singapore Flats - Resale Price Prediction]")


town = st.selectbox('Town', options=df['town'].unique())  
flat_type = st.selectbox('Flat Type', options=df['flat_type'].unique())  
block = st.selectbox('Block',options=df['block'].unique())
street_name = st.selectbox('Street Name',options=df['street_name'].unique())
storey_range = st.selectbox('Storey Range', options=df['storey_range'].unique()) 
flat_model = st.selectbox('Flat Model', options=df['flat_model'].unique())  
floor_area_sqm = st.number_input('Floor Area (sqm)', value=30.0 ,min_value=25.0 , max_value=370.0)
lease_commence_date = st.slider('Lease Commence Date', df['lease_commence_date'].min(),df['lease_commence_date'].max(),df['lease_commence_date'].min())
year_sold = st.slider('Year Sold', df['year'].min(),df['year'].max(),df['year'].min())
remaining_lease = 99 - (year_sold - lease_commence_date)

with open('label_encoders.pkl', 'rb') as enc_file:
        label_encoders = pickle.load(enc_file)


town_encoded = label_encoders['town'].transform([town])[0]
flat_type_encoded = label_encoders['flat_type'].transform([flat_type])[0]
block_encoded = label_encoders['block'].transform([block])[0]
street_name_encoded = label_encoders['street_name'].transform([street_name])[0]
storey_range_encoded = label_encoders['storey_range'].transform([storey_range])[0]
flat_model_encoded = label_encoders['flat_model'].transform([flat_model])[0]

my_input = [
    town_encoded,
    flat_type_encoded,
    block_encoded,
    street_name_encoded,
    storey_range_encoded,
    flat_model_encoded,
    floor_area_sqm,
    lease_commence_date,
    year_sold,
    remaining_lease
]

my_input = [my_input]

if st.button('Predict'):
    with open('Resale_price Prediction.pkl', 'rb') as ft:
        model = pickle.load(ft)

    prediction = model.predict(my_input)
    st.write(f"Selling Price: ${prediction[0]:,.2f}")
