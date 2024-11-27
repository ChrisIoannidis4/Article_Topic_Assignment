import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
from bertopic import BERTopic
from utils import preprocess_text
import seaborn as sns



trained_model_path= 'models/bertopic_model'


#Load and combine data with source flag and convert datetime format
insta_data = pd.read_csv('data/instagram.csv')
fb_data = pd.read_csv('data/facebook.csv')
insta_data['source'] = 'Instagram'
fb_data['source'] = 'Facebook'
combined_data = pd.concat([insta_data, fb_data]).reset_index(drop=True)
combined_data['date'] = pd.to_datetime(combined_data['date'])


def plot_views_per_post_per_topic(df_to_plot, model_path):

    # Load the trained BERTopic model
    model_path = model_path
    topic_model = BERTopic.load(model_path)

    #Apply the text cleaning/preprocessing
    df_to_plot['clean_text'] = df_to_plot['text'].apply(preprocess_text)

    #extract topics for all the texts with the loaded trained mode-with the most prominent keyword assigned
    #also filter out whichever is assigned to "-1" aka noise data
    topics, probabilities = topic_model.transform(df_to_plot['clean_text'])
    df_to_plot['topic'] = topics
    df_to_plot['topic_name'] = df_to_plot['topic'].apply(lambda t: topic_model.get_topic(t)[0][0] if t != -1 else 'Unknown')
    df_to_plot = df_to_plot[df_to_plot['topic_name'] != 'Unknown']


    posts_per_topic = df_to_plot.groupby('topic_name').size().reset_index(name='post_count') # number of posts per topic
    views_per_topic = df_to_plot.groupby('topic_name')['views_count'].sum().reset_index() # total views per topic
    #merge the two above dfs on topic_name and calculate (views/posts)/topic
    topic_stats = pd.merge(posts_per_topic, views_per_topic, on='topic_name')
    topic_stats['avg_views_per_post'] = topic_stats['views_count'] / topic_stats['post_count']
    topic_stats = topic_stats.sort_values(by='avg_views_per_post', ascending=False) #sorting 

    # Plot the average views per post per topic with applied styling
    plt.figure(figsize=(14, 8))
    bars = plt.barh(topic_stats['topic_name'], topic_stats['avg_views_per_post'], color='teal', edgecolor='black')
    plt.xlabel('Average Views per Post', fontsize=14, labelpad=15)
    plt.ylabel('Topic', fontsize=14, labelpad=15)
    plt.title('Average Views per Post per Topic', fontsize=20, pad=20, weight='bold')

    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}')) #for regular format numbers
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)

    #value labels for bars
    for bar in bars:
        plt.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                f'{int(bar.get_width()):,}', 
                va='center', ha='left', fontsize=12, color='black')

    plt.tight_layout()
    plt.savefig('output/avg_views_per_topic.jpg', format='jpg', dpi=300)
    plt.show()


plot_views_per_post_per_topic(combined_data, trained_model_path)
