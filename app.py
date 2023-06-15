import pandas as pd
import numpy as np
import streamlit as st 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from goose3 import Goose
import plotly.express as px 
import matplotlib.pyplot as plt
import dotenv
from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv('HUGGINGFACE_API'))


st.title('Sentiment Analysis Tool')

st.subheader('Vader Sentiment')

text = st.text_input('Enter a comment')
click=st.button('Compute')

print(text)

def senti(text):
    obj=SentimentIntensityAnalyzer()
    senti_dict=obj.polarity_scores(text)
    print(senti_dict)
    if senti_dict['compound']>=0.05:
        st.write("😁 Positive")
    elif senti_dict['compound']<=-0.05:
        st.write("😥 Negative")
    else:
        st.write("🙂 Neutral")
        


st.header("Hate speech classification")

import requests

API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
headers = {"Authorization": "Bearer " + st.secrets["HUGGINGFACE_API"]}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()



def senti_class(text):
    print(text)
    output=query({"inputs":str(text)})
    result={}
    if text:
        for data in output:
            print(data)
    st.write(output)
    return output

def viz(o):
    labels = [item['label'] for item in o[0]]
    scores = [item['score'] for item in o[0]]

    fig = px.bar(x=labels, y=scores)

    fig.update_layout(
        title="Emotional Scores",
        xaxis_title="Emotion",
        yaxis_title="Score"
    )

    st.plotly_chart(fig)

if click:
    senti(text)
    o=senti_class(text)
    viz(o)


# output = query({
# 	"inputs": "I hate you. I dont love you",
# })