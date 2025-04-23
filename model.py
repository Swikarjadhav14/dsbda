import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import re
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakeNewsModel:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading punkt tokenizer...")
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading stopwords...")
            nltk.download('stopwords', quiet=True)
            
        # Load the pre-trained model if available
        try:
            if os.path.exists('fake_news_model.pkl'):
                self.model = joblib.load('fake_news_model.pkl')
                logger.info("Loaded pre-trained model successfully")
            else:
                logger.info("No pre-trained model found. Training new model...")
                self.train_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Training new model...")
            self.train_model()

    def train_model(self):
        try:
            # Load the datasets
            true_df = pd.read_csv('dataset/True.csv')
            false_df = pd.read_csv('dataset/Fake.csv')

            # Combine both datasets
            true_df['label'] = 1  # True news label
            false_df['label'] = 0  # Fake news label

            data = pd.concat([true_df[['title', 'text', 'label']], false_df[['title', 'text', 'label']]])

            # Split the data
            X = data[['title', 'text']].apply(lambda x: ' '.join(x), axis=1)  # Combine title and text
            y = data['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Build the model pipeline
            self.model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())

            # Train the model
            self.model.fit(X_train, y_train)
            logger.info("Model trained successfully")

            # Save the trained model
            joblib.dump(self.model, 'fake_news_model.pkl')
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict(self, data):
        try:
            # Make prediction using the trained model
            return self.model.predict(data['title'] + ' ' + data['text'])
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            raise

    def predict_with_confidence(self, data):
        try:
            # Get prediction probabilities
            probs = self.model.predict_proba(data['title'] + ' ' + data['text'])
            
            # Get the prediction (0 or 1)
            prediction = self.model.predict(data['title'] + ' ' + data['text'])
            
            # Calculate confidence score (maximum probability)
            confidence = np.max(probs, axis=1)[0]
            
            return prediction[0], confidence
        except Exception as e:
            logger.error(f"Error in predict_with_confidence: {str(e)}")
            raise

    def analyze_text(self, title, text):
        try:
            # Combine title and text for analysis
            combined_text = f"{title} {text}"
            
            # Basic text statistics
            word_count = len(word_tokenize(combined_text))
            sentence_count = len(nltk.sent_tokenize(combined_text))
            
            # Sentiment analysis
            blob = TextBlob(combined_text)
            sentiment = blob.sentiment
            
            # Extract named entities
            entities = self._extract_entities(combined_text)
            
            # Analyze writing style
            style_analysis = self._analyze_writing_style(combined_text)
            
            # Analyze source reliability indicators
            reliability_indicators = self._analyze_reliability_indicators(combined_text)
            
            # Analyze content consistency
            consistency_analysis = self._analyze_content_consistency(combined_text)
            
            return {
                'basic_stats': {
                    'word_count': word_count,
                    'sentence_count': sentence_count,
                    'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0
                },
                'sentiment': {
                    'polarity': sentiment.polarity,
                    'subjectivity': sentiment.subjectivity
                },
                'entities': entities,
                'writing_style': style_analysis,
                'reliability_indicators': reliability_indicators,
                'content_consistency': consistency_analysis
            }
        except Exception as e:
            logger.error(f"Error in analyze_text: {str(e)}")
            # Return a basic analysis if there's an error
            return {
                'basic_stats': {
                    'word_count': len(combined_text.split()),
                    'sentence_count': len(combined_text.split('.')),
                    'avg_sentence_length': len(combined_text.split()) / len(combined_text.split('.')) if len(combined_text.split('.')) > 0 else 0
                },
                'sentiment': {
                    'polarity': 0,
                    'subjectivity': 0
                },
                'entities': {
                    'dates': [],
                    'numbers': [],
                    'proper_nouns': []
                },
                'writing_style': {
                    'unique_words_ratio': 0,
                    'stop_words_ratio': 0,
                    'most_common_words': []
                },
                'reliability_indicators': {
                    'has_quotes': False,
                    'has_links': False,
                    'has_numbers': False,
                    'has_dates': False
                },
                'content_consistency': {
                    'has_contradictions': False,
                    'contradictions': []
                }
            }

    def _extract_entities(self, text):
        # Simple entity extraction (can be enhanced with NER)
        # Look for common patterns like dates, numbers, and proper nouns
        dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
        numbers = re.findall(r'\d+', text)
        proper_nouns = [word for word in word_tokenize(text) if word[0].isupper()]
        
        return {
            'dates': dates,
            'numbers': numbers,
            'proper_nouns': proper_nouns
        }

    def _analyze_writing_style(self, text):
        # Analyze writing style characteristics
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        
        # Calculate word frequency
        word_freq = Counter(words)
        
        # Remove stop words for content word analysis
        content_words = [word for word in words if word not in stop_words]
        content_word_freq = Counter(content_words)
        
        return {
            'unique_words_ratio': len(set(words)) / len(words) if words else 0,
            'stop_words_ratio': len([w for w in words if w in stop_words]) / len(words) if words else 0,
            'most_common_words': content_word_freq.most_common(5)
        }

    def _analyze_reliability_indicators(self, text):
        # Analyze indicators of reliability
        has_quotes = '"' in text or "'" in text
        has_links = 'http' in text or 'www' in text
        has_numbers = bool(re.search(r'\d', text))
        has_dates = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text))
        
        return {
            'has_quotes': has_quotes,
            'has_links': has_links,
            'has_numbers': has_numbers,
            'has_dates': has_dates
        }

    def _analyze_content_consistency(self, text):
        # Analyze content consistency
        sentences = nltk.sent_tokenize(text)
        
        # Check for contradictory statements (simplified)
        contradictions = []
        for i in range(len(sentences)-1):
            if self._check_contradiction(sentences[i], sentences[i+1]):
                contradictions.append((sentences[i], sentences[i+1]))
        
        return {
            'has_contradictions': len(contradictions) > 0,
            'contradictions': contradictions
        }

    def _check_contradiction(self, sent1, sent2):
        # Simple contradiction detection (can be enhanced)
        # This is a basic implementation that looks for opposite sentiment
        blob1 = TextBlob(sent1)
        blob2 = TextBlob(sent2)
        
        # If one sentence is positive and the other is negative, it might be a contradiction
        return (blob1.sentiment.polarity > 0 and blob2.sentiment.polarity < 0) or \
               (blob1.sentiment.polarity < 0 and blob2.sentiment.polarity > 0)
