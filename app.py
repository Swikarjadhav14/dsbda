from flask import Flask, render_template, request, jsonify
from model import FakeNewsModel
import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize model
try:
    model = FakeNewsModel()
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template('index.html', error="Model not initialized properly. Please check the server logs.")

        title = request.form['title']
        text = request.form['text']
        
        if not title or not text:
            return render_template('index.html', error="Please provide both title and text.")

        data = pd.DataFrame([[title, text]], columns=['title', 'text'])
        
        # Get prediction and confidence score
        prediction, confidence = model.predict_with_confidence(data)
        
        # Get detailed analysis
        analysis = model.analyze_text(title, text)
        
        # Generate detailed explanation
        explanation = generate_detailed_explanation(prediction, confidence, analysis)
        
        result = 'True News' if prediction == 1 else 'Fake News'
        
        return render_template('index.html', 
                             prediction_text=f'The news is: {result}',
                             confidence_score=confidence,
                             analysis=analysis,
                             explanation=explanation)
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        return render_template('index.html', error="An error occurred while processing your request. Please try again.")

def generate_detailed_explanation(prediction, confidence, analysis):
    try:
        explanation = {
            'credibility_score': confidence * 100,
            'key_factors': [],
            'writing_style_analysis': [],
            'reliability_indicators': [],
            'content_analysis': []
        }
        
        # Analyze writing style
        if analysis['writing_style']['unique_words_ratio'] > 0.6:
            explanation['writing_style_analysis'].append({
                'factor': 'Vocabulary Diversity',
                'score': 'High',
                'description': 'The article uses a diverse vocabulary, which is often a sign of well-researched content.'
            })
        else:
            explanation['writing_style_analysis'].append({
                'factor': 'Vocabulary Diversity',
                'score': 'Low',
                'description': 'The article uses repetitive language, which might indicate rushed or unverified content.'
            })
        
        # Analyze reliability indicators
        reliability_score = 0
        if analysis['reliability_indicators']['has_quotes']:
            reliability_score += 1
            explanation['reliability_indicators'].append({
                'factor': 'Source Attribution',
                'score': 'Present',
                'description': 'The article includes direct quotes, which helps verify the information.'
            })
        
        if analysis['reliability_indicators']['has_links']:
            reliability_score += 1
            explanation['reliability_indicators'].append({
                'factor': 'External References',
                'score': 'Present',
                'description': 'The article includes links to external sources, allowing for fact verification.'
            })
        
        if analysis['reliability_indicators']['has_dates']:
            reliability_score += 1
            explanation['reliability_indicators'].append({
                'factor': 'Temporal Context',
                'score': 'Present',
                'description': 'The article includes specific dates, providing temporal context.'
            })
        
        # Analyze content consistency
        if analysis['content_consistency']['has_contradictions']:
            explanation['content_analysis'].append({
                'factor': 'Content Consistency',
                'score': 'Low',
                'description': 'The article contains contradictory statements, which may indicate unreliable information.'
            })
        else:
            explanation['content_analysis'].append({
                'factor': 'Content Consistency',
                'score': 'High',
                'description': 'The article maintains consistent information throughout.'
            })
        
        # Analyze sentiment
        sentiment = analysis['sentiment']
        if abs(sentiment['polarity']) > 0.5:
            explanation['content_analysis'].append({
                'factor': 'Emotional Tone',
                'score': 'High',
                'description': 'The article uses strong emotional language, which may affect objectivity.'
            })
        else:
            explanation['content_analysis'].append({
                'factor': 'Emotional Tone',
                'score': 'Balanced',
                'description': 'The article maintains a balanced and objective tone.'
            })
        
        # Add key factors based on analysis
        if prediction == 1:  # True News
            explanation['key_factors'].extend([
                'Well-structured content with proper attribution',
                'Consistent information throughout the article',
                'Balanced and objective reporting style'
            ])
        else:  # Fake News
            explanation['key_factors'].extend([
                'Potential inconsistencies in reporting',
                'Limited source attribution',
                'May contain emotional or biased language'
            ])
        
        return explanation
    except Exception as e:
        logger.error(f"Error in generate_detailed_explanation: {str(e)}")
        return {
            'credibility_score': 0,
            'key_factors': ['Error in analysis'],
            'writing_style_analysis': [],
            'reliability_indicators': [],
            'content_analysis': []
        }

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return render_template('index.html', error="An internal server error occurred. Please try again later.")

if __name__ == "__main__":
    # Ensure NLTK data is downloaded
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
    
    app.run(debug=True)
