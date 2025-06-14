# News Authenticity Analyzer

A Flask-based web application that analyzes news articles for authenticity using machine learning and natural language processing.

## Features

- News authenticity detection
- Detailed analysis of writing style
- Sentiment analysis
- Content consistency checking
- Source reliability indicators
- Interactive visualization of results

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd news-authenticity-analyzer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```bash
python download_nltk_data.py
```

5. Run the application:
```bash
python app.py
```

## Deployment

### Heroku Deployment

1. Create a Heroku account and install the Heroku CLI
2. Login to Heroku:
```bash
heroku login
```

3. Create a new Heroku app:
```bash
heroku create your-app-name
```

4. Deploy to Heroku:
```bash
git push heroku main
```

### Other Platforms

The application can be deployed on any platform that supports Python web applications (e.g., AWS, Google Cloud, DigitalOcean).

## Project Structure

```
.
├── app.py              # Main Flask application
├── model.py            # ML model and analysis logic
├── requirements.txt    # Python dependencies
├── Procfile           # Heroku deployment configuration
├── download_nltk_data.py  # NLTK data download script
├── templates/         # HTML templates
│   └── index.html     # Main template
└── dataset/          # Training data
    ├── True.csv
    └── Fake.csv
```

## Requirements

- Python 3.8+
- Flask
- scikit-learn
- NLTK
- TextBlob
- Other dependencies listed in requirements.txt

## License

MIT License
