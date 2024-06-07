import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from textblob import TextBlob

nltk.download('punkt')
nltk.download('wordnet')

class TextMining:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text = self.load_text()
        self.normalized_text = None
        self.tokens = None
        self.stemmed_tokens = None
        self.lemmatized_tokens = None
        self.vectorizer = None
        self.vectorized_data = None
        self.lda_model = None
        self.entities = None
        self.sentiment = None

    def load_text(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def normalize_text(self):
        text = self.text.lower()  # Convert to lowercase.
        text = re.sub(r'\bmr\b', 'mister', text)
        text = re.sub(r'\bmrs\b', 'misses', text)
        text = re.sub(r'\bdr\b', 'doctor', text)
        text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation.
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces.
        self.normalized_text = text
        return self.normalized_text

    def tokenize_text(self):
        self.tokens = word_tokenize(self.normalized_text)
        return self.tokens

    def stem_text(self):
        stemmer = PorterStemmer()
        self.stemmed_tokens = [stemmer.stem(token) for token in self.tokens]
        return self.stemmed_tokens

    def lemmatize_text(self):
        lemmatizer = WordNetLemmatizer()
        self.lemmatized_tokens = [lemmatizer.lemmatize(token) for token in self.tokens]
        return self.lemmatized_tokens

    def vectorize_text(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.vectorized_data = self.vectorizer.fit_transform([' '.join(self.lemmatized_tokens)])
        return self.vectorized_data

    def perform_ner(self):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(self.text)
        self.entities = [(entity.text, entity.label_) for entity in doc.ents]
        return self.entities

    def analyze_sentiment(self):
        blob = TextBlob(self.text)
        self.sentiment = blob.sentiment
        return self.sentiment

    def topic_modeling(self, n_topics=5):
        self.lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        self.lda_model.fit(self.vectorized_data)
        topics = {}
        for i, topic in enumerate(self.lda_model.components_):
            topics[i] = [self.vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]]
        return topics

    def run_all(self):
        self.normalize_text()
        self.tokenize_text()
        self.stem_text()
        self.lemmatize_text()
        self.vectorize_text()
        ner_results = self.perform_ner()
        sentiment_results = self.analyze_sentiment()
        topic_results = self.topic_modeling()
        return {
            "NER": ner_results,
            "Sentiment": sentiment_results,
            "Topics": topic_results
        }

# Usage.
file_path = '01 Harry Potter and the Sorcerers Stone.txt'
tm = TextMining(file_path)
results = tm.run_all()

# Display results.
print("Named Entities:")
print(results['NER'])
print("\nSentiment Analysis:")
print(f"Polarity: {results['Sentiment'].polarity}, Subjectivity: {results['Sentiment'].subjectivity}")
print("\nTopics:")
for topic, words in results['Topics'].items():
    print(f"Topic {topic}: {', '.join(words)}")
