## Christos Ioannidis, MSc AI

### Train BERTopic model for topic assignment in football articles:\\
### bertopic_model.py

I choose BERTopic because it has been characterized as a state of the art topic assignment and summarization method and is used across
both research and industry level, according to my knowledge.

I first preprocess the text, so that the topic assignment does not focus on noise or stopwords, or common terms like 'football' or 'player' but focuses more on the essence of the text. This is done in the 'utils.py' text, so that it is kept as a process that can be applied whenever the model or any text processing is required 
There I (as is detailed in the code):

- normalize the text so that it is all in lowercase
- remove special characters
- remove stopwords according the domain-specific ones we added by expertise or by trial and error on what was classified as noise (aka. what was falling on the -1 category after the training of the model in iterated runs)
- reduces words to their base or root form with NLTK's 'WordNetLemmatizer' (so 'scoring' or 'scores' would transform to 'score')

The embedding model Ι choose is "all-mpnet-base-v2" that converts the text into high-dimensional vectors to input to the BERTopic model.
The choice is justified by both trial and error as well as the fact that despite it's more advanced than e.g. all-MiniLM-L6-v2, which is needed because the texts in this article are related/in the same domain, it still remains an efficient choice. To my knowledge it captures semantic relationships well and is efficient for clustering / topic assignment.

In the model training, n_gram range is (1,3) to keep track of unigrams, bigrams, and trigrams, and min_topic_size is set to 8 after fine tuning and trial and error from my side - or in other words, after studying the results with different sizes.

I train the model and add the assignment of the topics as seperate columns in the fb_data dataframe to output as a file in the same directory in the folder 'output' with the name "facebook_news_topic". The topic_name column shows the assigned topic for the article.

The trained model is also saved in the "models" folder for future predictions.

Also, the topic names along with the category(-1 corresponds to noise or very common words") and their popularity are printed.

The topic assignment seems coherent, although the number of assignments to the class -1 remains high. With more time, further analysis and fine tuning could reduce that.

### Integration to a pipeline 
### predict_topic.py

Having saved the model (in other words after having run bertopic_model.py), the article name can be passed as argument (e.g. the PythonOperator/Databricks operators could have it as a passed argument) and output the assigned topic.
After the article name is obtained, the preprocessing, model-loading, and predicting steps could be seen in asgn_2_bonus.py folder.

In my file, the user can pass the news title in the variable article_name in line 13. Otherwise, in the aforementioned setting, a sys.argv command would work. The predict_article_topic function takes the name of the article, specified before, and the path of the model saved in the directory as inputs, and then proceeds to preprocess the data (from the utils.preprocess_text method)

Then, the outputs that are printed are the Predicted Topic Number (categorical, corresponds to the names that were extracted in the previous setp) and the Topic Keywords that could be used as tags to this article. 

The plot clearly shows that Olympics related posts are the most popular - timing is important for this- through the keywords olympic, and paris.
Transfer and scoreline news follow up.
The billions category could have to do with takeovers, transfers and other types of articles.


### Visualization views per topic:

I combined data from both sources (although facebook has been used to train the model, this is just to showcase the utility of the option).
I applied the same preprocessing as before, saved in utils. Then the topics are predicted through the model that has been loaded.
Then by some dataframe manipulation I can obtain the necessary results, which are displayed and saved in the same directory in the output folder with the filename avg_views_per_topic.jpg.

