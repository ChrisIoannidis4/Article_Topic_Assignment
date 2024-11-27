import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
from bertopic import BERTopic
from utils import preprocess_text

import pandas as pd
import os
from bertopic import BERTopic
from utils import preprocess_text

article_name = "Juventus Boss Thiago Motta confirms that Federico Chiesa is not part of his plans for the upcoming season."
model_path='models/bertopic_model'
topic_model = BERTopic.load(model_path)


def predict_article_topic(article_name):


    clean_text = preprocess_text(article_name)
    
    # Predict the topic using the trained model
    topics, probabilities = topic_model.transform([clean_text])
    
    # Get the most relevant topic and its keywords
    topic_number = topics[0]
    topic_keywords = topic_model.get_topic(topic_number)
    
    return topic_number, topic_keywords


topic_number, topic_keywords = predict_article_topic(article_name)


print(f"Predicted Topic Number: {topic_number}")
print(f"Topic Keywords: {', '.join([keyword for keyword, _ in topic_keywords])}")