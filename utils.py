
import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords

# Download the necessary NLTK data files for the preprocessing
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

general_stopwords = set(stopwords.words('english'))
football_stopwords = {'club', 'season', 'team', 'player', 'league', 'match', 'game', 'goal', 'football'}
all_stopwords = general_stopwords.union(football_stopwords)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove short words
    text = re.sub(r'\b\w{1,2}\b', '', text)
    # Remove stopwords, including domain-specific ones
    text = ' '.join([word for word in text.split() if word not in all_stopwords])
    # Lemmatize the text
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text