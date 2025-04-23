import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_save_model():
    try:
        logger.info("Loading datasets...")
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

        logger.info("Building model pipeline...")
        # Build the model pipeline
        model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())

        logger.info("Training model...")
        # Train the model
        model.fit(X_train, y_train)

        logger.info("Evaluating model...")
        # Evaluate the model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        logger.info(f"Train accuracy: {train_score:.4f}")
        logger.info(f"Test accuracy: {test_score:.4f}")

        logger.info("Saving model...")
        # Save the trained model
        joblib.dump(model, 'fake_news_model.pkl')
        logger.info("Model saved successfully!")

    except Exception as e:
        logger.error(f"Error in train_and_save_model: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model() 