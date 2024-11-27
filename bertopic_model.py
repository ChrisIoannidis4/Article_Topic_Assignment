import pandas as pd 
import pandasql as psql
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import os
from utils import preprocess_text

fb_data = pd.read_csv('data/facebook.csv')

# we load the preprocessing steps we created in utils
fb_data['clean_text'] = fb_data['text'].apply(preprocess_text)
# representation_model = [KeyBERTInspired(), MaximalMarginalRelevance(diversity=0.7)]

#choose embedding, and train our model
embedding_model = SentenceTransformer("all-mpnet-base-v2")
topic_model = BERTopic(embedding_model=embedding_model, n_gram_range=(1, 3), min_topic_size=8)
topics, probabilities = topic_model.fit_transform(fb_data['clean_text'])

# extract topics and add to initial df
topics_overview = topic_model.get_topic_info()
document_info = topic_model.get_document_info(fb_data['clean_text'])
fb_data['topic'] = document_info['Topic']
fb_data['topic_name'] = document_info['Name']

print(topics_overview)

#save the df as csv
fb_data.to_csv('output/facebook_news_topics.csv', index=False)

#save the trained model for further use (for bonus assignment and assignment 3)
model_path = 'models/bertopic_model'
os.makedirs('models', exist_ok=True)
topic_model.save(model_path)
