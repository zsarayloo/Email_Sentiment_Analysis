####################################################################################
####################################################################################
####################################################################################
#### Author : Zahra Sarayloo #######################################################
#### Description:  This code is written to analyze email from Eron dataset #########
####################################################################################
####################################################################################


import numpy as np
import os
import glob
import pandas as pd
from email import policy
from email.parser import BytesParser
import re
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import joblib
import pickle
from transformers import pipeline
import spacy
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer



############################################## FUNCTIONS #############################


########## Parsing emails 


def email_parser(email_dir):

    with open(email_dir,'rb') as file:
        email_content   =   file.read()

    email       =   BytesParser(policy = policy.default).parsebytes(email_content)

    # Extract email_body

    email_body  =   ""

    if email.is_multipart():
        
        for part in email.iter_parts():
            
            if part.get_content_type() == "text/plain":
                
                email_body += part.get_payload(decode = True).decode(part.get_content_charset())
    else: 
        email_body = email.get_payload(decode = True).decode(email.get_content_charset())


    return {'Sender': email['from'],'Receiver': email['to'],'Subject': email['subject'],'date': email['date'], 'body': email_body}


########## Load_Data

def collect_mails(main_directory):
    email_data = []
    Sent_mails = glob.glob(os.path.join(main_directory,'**','sent'),recursive = True)
    
    for sent_dir in Sent_mails:
        files = glob.glob(os.path.join(sent_dir,'*.'))
        
        for File in files:

            email_data.append(email_parser(File))
                
    return pd.DataFrame(email_data)

########## Data_Cleaning

def clean_data(data):


    data = re.sub(r'\n',' ',data) # Replace new line with space
    data = re.sub(r'\W','  ',data) # Remove non-words characters
    data = re.sub(r'\s+',' ',data) # Remove extra spaces
    data = re.sub(r'[^\w\s]','',data)
    data = re.sub(r'[0-9]+','',data)
    data = data.lower().strip() # Convert to lower case and strip white space
    data = re.sub('re:', '', data) 
    data = re.sub('-', '', data)
    data = re.sub('_', '', data)
    data =re.sub('\[[^]]*\]', '', data)

    p = re.compile(r'<.*?>')
    
    data = re.sub(r"\'ve", " have ", data)
    data = re.sub(r"can't", "cannot ", data)
    data = re.sub(r"n't", " not ", data)
    data = re.sub(r"I'm", "I am", data)
    data = re.sub(r" m ", " am ", data)
    data = re.sub(r"\'re", " are ", data)
    data = re.sub(r"\'d", " would ", data)
    data = re.sub(r"\'ll", " will ", data)
    data = re.sub('forwarded by phillip k allenhouect on    pm', '',data)
    data = re.sub(r"httpitcappscorpenroncomsrrsauthemaillinkaspidpage", "", data)
    data = p.sub('', data)
    
    if 'forwarded by:' in data:
        data = data.split('subject')[1]
        data = data.strip()
    
    return data
    
############  Extract Sentiment based on scores

def Extract_Sentimnt(compound):
    if compound>= 0.05:
        return "Positive"
        
    elif compound<= -0.05:
        return "Negative"
        
    else:
        return "Neutral"

############ Checking the emails that are related to oil & gas

def check_oil_gas_relation(data):

    Keyword_oil_gas_related =   ['oil', 'gas', 'energy', 'pipeline', 'drilling', 'petroleum', 'refinery']
    
    return any(keyword in data for keyword in Keyword_oil_gas_related)

########### Function to extract entities from text

def extract_entities(text,NER_model):

    doc         =   NER_model(text)
    entities    =   [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]
    
    return entities


###################################### MAIN BODY ##################
######################################           ##################

### Define Parameters

Max_Vocab_size  =   5000

Stop_Words      =   'english'

N_Components    =   10

random_Seed     =   2024

Num_top_word    =   10




### Load the Enron Data_Set from current directory and Create a DataFrame

if os.path.exists('email_df.pkl'):
    
    email_DF    = pd.read_pickle('email_df.pkl')
else:

    current_dir =   os.getcwd()
    
    email_DF    =   collect_mails(current_dir)
    
    ### Pre-Process Data_set ##############

    email_DF['Pre_Processed_Body']  =   email_DF['body'].apply(clean_data)
    
    email_DF.to_pickle('email_df.pkl')



##### Unsupervised Sentiment analysis by vaderSentiment


sentiment_analyser = SentimentIntensityAnalyzer()


email_DF['scores']      =   email_DF['Pre_Processed_Body'].apply(lambda review: sentiment_analyser.polarity_scores(review))

email_DF['compound']    =   email_DF['scores'].apply(lambda score_dict: score_dict['compound'])

email_DF['Sentiment']   =   email_DF['compound'].apply(Extract_Sentimnt)


### Sentiment Analysis by Topic Extraction
'''
# Model 1 based on BERT (transformers)

if os.path.exists('Bert_model.pkl'):

    with open('Bert_model.pkl','rb') as B:
        
        Bert_model  =   pickle.load(B)
else:

    Bert_model      =    BERTopic()  # build a BERT model

    bert_topic, _   =   Bert_model.fit_transform(email_DF['Pre_Processed_Body'].tolist())   # fit the model
    
    with open('Bert_model.pkl','wb') as B:
        
        pickle.dump(Bert_model, B)


email_DF['bert_topic']  =   bert_topic   # add extracted topics based on bert_model

# Another method based on Transformers

# load pre-trained sentiment analysis pipeline

Sentiment_analysis_model    =   pipeline("sentiment-analysis")

email_DF['transformer_Sentiment']       =   email_DF['Pre_Processed_Body'].apply(lambda x: Sentiment_analysis_model(x)[0]['label'])

'''

# Model 2 based on Non-Negative Matrix Factorization (NMF)

if os.path.exists('NMF_model.pkl'):

    NMF_model       =   joblib.load('NMF_model.pkl')
    TF_Vectorizer   =   joblib.load('TF_Vectorizer.pkl')
    

else:

    
    TF_Vectorizer   =   TfidfVectorizer(max_features = Max_Vocab_size, stop_words = Stop_Words)

    Transformed_Doc =   TF_Vectorizer.fit_transform(email_DF['Pre_Processed_Body'])

    NMF_model       =   NMF(n_components = N_Components, random_state = random_Seed) # build model

    Topic_matrix    =   NMF_model.fit_transform(Transformed_Doc) # Topic Matrix

    joblib.dump(NMF_model,'NMF_model.pkl')
    joblib.dump(TF_Vectorizer,'TF_Vectorizer.pkl')


# Get top words for each topic
Transformed_Doc     =   TF_Vectorizer.fit_transform(email_DF['Pre_Processed_Body'])

Topic_matrix        =   NMF_model.fit_transform(Transformed_Doc) # Topic Matrix

Feature_names       =   TF_Vectorizer.get_feature_names_out()

Topic_word_matrix   =   NMF_model.components_  # Topic-term matrix

num_top_words       =   Num_top_word  # Number of top words to display per topic

Topic_top_words     =   []

for topic_idx, topic in enumerate(Topic_word_matrix):

    top_indices =   topic.argsort()[:-num_top_words - 1:-1]
    
    top_words   =   [Feature_names[i] for i in top_indices]
    
    Topic_top_words.append(', '.join(top_words))


NMF_topics              =   Topic_matrix.argmax(axis=1)

email_DF['NMF_topic']   =   NMF_topics

email_DF['Top_Words']   =   [Topic_top_words[idx] for idx in NMF_topics] 






##### Find the emails that are related to Oil & Gas

email_DF['Is_to_Oil_Gas']   =   email_DF['Pre_Processed_Body'].apply(check_oil_gas_relation)



##### Find the person name and organization in emails

NER_model = spacy.load("en_core_web_sm") # load pre-trained Spacy model for NER

# Apply NER to emails related to oil & gas
email_DF['Entities'] = email_DF.apply(lambda row: extract_entities(row['Pre_Processed_Body'],NER_model) if row['Is_to_Oil_Gas'] else [], axis=1)

# Display results

print(email_DF[['Subject', 'NMF_topic', 'Top_Words','Sentiment','Is_to_Oil_Gas', 'Entities']].head())

email_DF.to_pickle('email_df.pkl')
joblib.dump(NMF_model,'NMF_model.pkl')



