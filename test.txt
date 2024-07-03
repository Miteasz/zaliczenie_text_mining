import re
import nltk
import logging
from collections import Counter
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from textblob import TextBlob

# Pobieranie zasobów NLTK.
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Konfiguracja loggera.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"analiza_{timestamp}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

class TextMining:
    """
    Klasa TextMining wykonuje różne operacje związane z eksploracją danych tekstowych, takie jak normalizacja tekstu, 
    tokenizacja, stemming, lematyzacja, wektoryzacja, rozpoznawanie nazwanych encji (NER), analiza sentymentu oraz modelowanie tematów.
    
    Atrybuty:
        file_path (str): Ścieżka do pliku tekstowego.
        text (str): Załadowany tekst z pliku.
        normalized_text (str): Znormalizowany tekst.
        tokens (list): Lista tokenów.
        stemmed_tokens (list): Lista stemowanych tokenów.
        lemmatized_tokens (list): Lista lematyzowanych tokenów.
        vectorizer (TfidfVectorizer): Obiekt TfidfVectorizer używany do wektoryzacji tekstu.
        vectorized_data (sparse matrix): Zwektoryzowane dane.
        lda_model (LatentDirichletAllocation): Model LDA do modelowania tematów.
        entities (list): Lista rozpoznanych encji.
        sentiment (Sentiment): Wyniki analizy sentymentu.
    """

    def __init__(self, file_path):
        """
        Inicjalizuje obiekt klasy TextMining i ładuje tekst z podanego pliku.

        Args:
            file_path (str): Ścieżka do pliku tekstowego.
        """
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
        logging.info("TextMining instance created.")

    def load_text(self):
        """
        Ładuje tekst z pliku.

        Returns:
            str: Załadowany tekst.
        """
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        logging.info(f"Loaded text from {self.file_path}.")
        return text

    def initial_analysis(self):
        """
        Przeprowadza wstępną analizę danych tekstowych, obliczając podstawowe parametry statystyczne.

        Returns:
            dict: Słownik zawierający parametry statystyczne tekstu.
        """
        word_count = len(self.text.split())
        unique_words = len(set(self.text.split()))
        average_word_length = sum(len(word) for word in self.text.split()) / word_count
        word_freq = Counter(self.text.lower().split())
        most_common_words = word_freq.most_common(3)
        main_character_counts = {
            'Harry': self.text.lower().count('harry'),
            'Ron': self.text.lower().count('ron'),
            'Hermione': self.text.lower().count('hermione')
        }
        stats = {
            "word_count": word_count,
            "unique_words": unique_words,
            "average_word_length": average_word_length,
            "most_common_words": most_common_words,
            "main_character_counts": main_character_counts
        }
        logging.info(f"Initial analysis completed: {stats}.")
        return stats

    def normalize_text(self):
        """
        Normalizuje tekst przez konwersję na małe litery, usunięcie interpunkcji i dodatkowych spacji.

        Returns:
            str: Znormalizowany tekst.
        """
        # Konwersja tekstu na małe litery.
        text = self.text.lower()
        # Zamiana 'mr' na 'mister'.
        text = re.sub(r'\bmr\b', 'mister', text)
        # Zamiana 'mrs' na 'misses'.
        text = re.sub(r'\bmrs\b', 'misses', text)
        # Zamiana 'dr' na 'doctor'.
        text = re.sub(r'\bdr\b', 'doctor', text)
        # Usunięcie interpunkcji.
        text = re.sub(r'[^a-z\s]', '', text)
        # Usunięcie dodatkowych spacji.
        text = re.sub(r'\s+', ' ', text).strip()
        # Zapisanie znormalizowanego tekstu.
        self.normalized_text = text
        logging.info("Text normalized.")
        return self.normalized_text

    def tokenize_text(self):
        """
        Tokenizuje znormalizowany tekst.

        Returns:
            list: Lista tokenów.
        """
        # Tokenizacja tekstu.
        self.tokens = word_tokenize(self.normalized_text)
        logging.info("Text tokenized.")
        return self.tokens

    def remove_stopwords(self):
        """
        Usuwa stop words z tokenów.

        Returns:
            list: Lista tokenów bez stop words.
        """
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in self.tokens if token not in stop_words]
        self.tokens = filtered_tokens
        logging.info("Stop words removed.")
        return self.tokens

    def stem_text(self):
        """
        Wykonuje stemming tokenów.

        Returns:
            list: Lista stemowanych tokenów.
        """
        # Utworzenie obiektu stemmera.
        stemmer = PorterStemmer()
        # Stemming tokenów.
        self.stemmed_tokens = [stemmer.stem(token) for token in self.tokens]
        logging.info("Text stemmed.")
        return self.stemmed_tokens

    def lemmatize_text(self):
        """
        Wykonuje lematyzację tokenów.

        Returns:
            list: Lista lematyzowanych tokenów.
        """
        # Utworzenie obiektu lematyzatora.
        lemmatizer = WordNetLemmatizer()
        # Lematyzacja tokenów.
        self.lemmatized_tokens = [lemmatizer.lemmatize(token) for token in self.tokens]
        logging.info("Text lemmatized.")
        return self.lemmatized_tokens

    def vectorize_text(self, max_features=1000):
        """
        Wektoryzuje tekst przy użyciu TF-IDF.

        Args:
            max_features (int): Maksymalna liczba cech do uwzględnienia.

        Returns:
            sparse matrix: Zwektoryzowane dane.
        """
        # Utworzenie obiektu TfidfVectorizer.
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        # Wektoryzacja tekstu.
        self.vectorized_data = self.vectorizer.fit_transform([' '.join(self.lemmatized_tokens)])
        logging.info("Text vectorized using TF-IDF.")
        return self.vectorized_data

    def perform_ner(self):
        """
        Wykonuje rozpoznawanie nazwanych encji (NER) przy użyciu spaCy.

        Returns:
            list: Lista rozpoznanych encji.
        """
        try:
            # Próba załadowania modelu spaCy.
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            # Pobieranie modelu jeżeli go nie ma.
            from spacy.cli import download
            download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')
        # Przetwarzanie tekstu za pomocą spaCy.
        doc = nlp(self.text)
        # Wyodrębnienie encji.
        self.entities = [(entity.text, entity.label_) for entity in doc.ents]
        logging.info("Named Entity Recognition performed.")
        return self.entities

    def analyze_sentiment(self):
        """
        Analizuje sentyment tekstu przy użyciu TextBlob.

        Returns:
            Sentiment: Wyniki analizy sentymentu (polaryzacja i subiektywność).
        """
        # Utworzenie obiektu TextBlob.
        blob = TextBlob(self.text)
        # Analiza sentymentu.
        self.sentiment = blob.sentiment
        logging.info("Sentiment analysis performed.")
        return self.sentiment

    def topic_modeling(self, n_topics=5):
        """
        Wykonuje modelowanie tematów przy użyciu Latent Dirichlet Allocation (LDA).

        Args:
            n_topics (int): Liczba tematów do wyodrębnienia.

        Returns:
            dict: Słownik tematów i ich najważniejszych słów.
        """
        # Utworzenie modelu LDA.
        self.lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        # Dopasowanie modelu do danych.
        self.lda_model.fit(self.vectorized_data)
        topics = {}
        for i, topic in enumerate(self.lda_model.components_):
            # Wyodrębnienie najważniejszych słów dla każdego tematu.
            topics[i] = [self.vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]]
        logging.info("Topic modeling performed using LDA.")
        return topics

    def run_all(self):
        """
        Uruchamia wszystkie operacje przetwarzania tekstu: wstępna analiza danych, normalizacja, tokenizacja, usunięcie stop words, stemming, lematyzacja, wektoryzacja, 
        NER, analiza sentymentu oraz modelowanie tematów.

        Returns:
            dict: Słownik zawierający wyniki wstępnej analizy danych, NER, analizy sentymentu i modelowania tematów.
        """
        # Wstępna analiza danych.
        initial_stats = self.initial_analysis()
        # Normalizacja tekstu.
        self.normalize_text()
        # Tokenizacja tekstu.
        self.tokenize_text()
        # Usunięcie stop words.
        self.remove_stopwords()
        # Stemming tokenów.
        self.stem_text()
        # Lematyzacja tokenów.
        self.lemmatize_text()
        # Wektoryzacja tekstu.
        self.vectorize_text()
        # Rozpoznawanie nazwanych encji.
        ner_results = self.perform_ner()
        # Analiza sentymentu.
        sentiment_results = self.analyze_sentiment()
        # Modelowanie tematów.
        topic_results = self.topic_modeling()
        logging.info("All text processing operations completed.")
        return {
            "Initial Analysis": initial_stats,
            "NER": ner_results,
            "Sentiment": sentiment_results,
            "Topics": topic_results
        }

# Użycie klasy.
file_path = '01 Harry Potter and the Sorcerers Stone.txt'
# Utworzenie obiektu TextMining.
tm = TextMining(file_path)
# Uruchomienie wszystkich operacji przetwarzania tekstu.
results = tm.run_all()

# Wyświetlenie wyników w konsoli oraz zapisanie do pliku.
with open(f"analiza_{timestamp}.txt", 'w', encoding='utf-8') as f:
    # Wyświetlenie i zapisanie wyników wstępnej analizy.
    print("Initial Analysis:")
    f.write("Initial Analysis:\n")
    for key, value in results['Initial Analysis'].items():
        print(f"{key}: {value}")
        f.write(f"{key}: {value}\n")

    # Wyświetlenie i zapisanie rozpoznanych encji.
    print("\nNamed Entities:")
    f.write("\nNamed Entities:\n")
    for entity in results['NER']:
        print(entity)
        f.write(f"{entity}\n")

    # Wyświetlenie i zapisanie wyników analizy sentymentu.
    print("\nSentiment Analysis:")
    f.write("\nSentiment Analysis:\n")
    print(f"Polarity: {results['Sentiment'].polarity}, Subjectivity: {results['Sentiment'].subjectivity}")
    f.write(f"Polarity: {results['Sentiment'].polarity}, Subjectivity: {results['Sentiment'].subjectivity}\n")

    # Wyświetlenie i zapisanie wyników modelowania tematów.
    print("\nTopics:")
    f.write("\nTopics:\n")
    for topic, words in results['Topics'].items():
        print(f"Topic {topic}: {', '.join(words)}")
        f.write(f"Topic {topic}: {', '.join(words)}\n")
