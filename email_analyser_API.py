####################################################################################
####################################################################################
####################################################################################
#### Author : Zahra Sarayloo #######################################################
#### Description:  This code is written to analyze email from Eron dataset #########
####################################################################################
####################################################################################

import os
import re
import pandas as pd
from email import policy
from email.parser import BytesParser
from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import joblib


app =   Flask(__name__)



########### Loading Pre_defined Models 

NER_model           =   spacy.load("en_core_web_sm") # load pre-trained spacy model for NER

sentiment_analyser  =   SentimentIntensityAnalyzer() # load sentiment_analyzer

NMF_model           =   joblib.load('NMF_model.pkl') # load NMF model

TF_Vectorizer       =   joblib.load('TF_Vectorizer.pkl') # load NMF tokenizer


########### Define Functions

##### Cleaning data function

def clean_data(data):

    data = re.sub(r'\n', ' ', data)
    data = re.sub(r'\W', '  ', data)
    data = re.sub(r'\s+', ' ', data)
    data = re.sub(r'[^\w\s]', '', data)
    data = re.sub(r'[0-9]+', '', data)
    data = data.lower().strip()
    data = re.sub('re:', '', data)
    data = re.sub('-', '', data)
    data = re.sub('_', '', data)
    data = re.sub('\[[^]]*\]', '', data)

    p = re.compile(r'<.*?>')
    data = re.sub(r"\'ve", " have ", data)
    data = re.sub(r"can't", " cannot ", data)
    data = re.sub(r"n't", " not ", data)
    data = re.sub(r"I'm", "I am", data)
    data = re.sub(r" m ", " am ", data)
    data = re.sub(r"\'re", " are ", data)
    data = re.sub(r"\'d", " would ", data)
    data = re.sub(r"\'ll", " will ", data)
    data = re.sub('forwarded by phillip k allenhouect on    pm', '', data)
    data = re.sub(r"httpitcappscorpenroncomsrrsauthemaillinkaspidpage", "", data)
    data = p.sub('', data)
    
    if 'forwarded by:' in data:
        data = data.split('subject')[1]
        data = data.strip()
    
    return data
##### Sentiment extraction based on score

def extract_sentiment(email_body):

    scores      =   sentiment_analyser.polarity_scores(email_body)
    compound    =   scores['compound']

    if compound >= 0.05:
        return "Positive"

    elif compound <= -0.05:
        return "Negative"

    else:
        return "Neutral"


##### Check the email is related to oil & gas

def check_oil_gas_relation(data):

    keyword_oil_gas_related =   ['oil', 'gas', 'energy', 'pipeline', 'drilling', 'petroleum', 'refinery']
    
    return any(keyword in data for keyword in keyword_oil_gas_related)

##### extraction of name of persons and organization from email

def extract_entities(text):
    doc = NER_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]

    return entities

##### Topic modeling

def nmf_topic_modeling(email_body):

    transformed_doc =   TF_Vectorizer.transform([email_body])
    topic_matrix    =   NMF_model.transform(transformed_doc)
    topic_idx       =   topic_matrix.argmax(axis=1)[0]
    
    return topic_idx



@app.route('/Analyse_email', methods=['POST'])


def Analyse_email():

    data        =   request.json
    raw_email   =    data['Email']

    # Parse the email

    email       =   BytesParser(policy=policy.default).parsebytes(raw_email.encode())
    
    email_body  =   ""

    if email.is_multipart():

        for part in email.iter_parts():

            if part.get_content_type() == "text/plain":

                email_body  +=  part.get_payload(decode=True).decode(part.get_content_charset())
    
    else:
        email_body = email.get_payload(decode=True).decode(email.get_content_charset())

    # Clean email body
    clean_body = clean_data(email_body)

    # Extract sentiment
    sentiment = extract_sentiment(clean_body)

    # Check if related to oil & gas
    is_oil_gas_related = check_oil_gas_relation(clean_body)

    # Extract entities if related to oil & gas
    entities = extract_entities(clean_body) if is_oil_gas_related else []

    # Get topic index
    topic_idx = nmf_topic_modeling(clean_body)

    # Prepare the output
    result = {
        'Sentiment': sentiment,
        'IsOilGasRelated': is_oil_gas_related,
        'Entities': entities,
        'TopicIndex': topic_idx
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

