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

st.subheader('HELLO USER')

text = st.text_input('Enter a comment')
click=st.button('Compute')

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
        

import requests

API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"

#headers = {"Authorization": "Bearer " + os.getenv('HUGGINGFACE_API')}
headers = {"Authorization": "Bearer " + st.secrets['HUGGINGFACE_API']}

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
    st.write("Emotion Table")
    print("this is output ")
    print(output)
    st.table(pd.DataFrame(output[0]))


    return output

def viz(o,text):
    obj=SentimentIntensityAnalyzer()
    senti_dict=obj.polarity_scores(text)
    sentidict=[]
    sentidict.append(senti_dict)
    st.write("Sentiment Table")
    st.table(pd.DataFrame(sentidict))
    labels = [item['label'] for item in o[0]]
    scores = [item['score'] for item in o[0]]
    labels.append('Neutral')
    labels.append('positive')
    labels.append('negative')
    scores.append(senti_dict['neu'])
    scores.append(senti_dict['pos'])
    scores.append(senti_dict['neg'])
    fig = px.bar(x=labels, y=scores)
    fig2 = px.pie(names=labels, values=scores)
    fig.update_layout(
        title="Sentiment Analysis",
        xaxis_title="Emotion",
        yaxis_title="Score"
    )

    st.plotly_chart(fig)
    st.plotly_chart(fig2)

if click:
    senti(text)
    o=senti_class(text)
    viz(o,text)


# output = query({
# 	"inputs": "I hate you. I dont love you",
# })