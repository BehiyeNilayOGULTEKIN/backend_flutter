
# Content Topic Analyzer API

This Flask API extracts and analyzes webpage text to identify main topics and categorize content based on predefined keyword lists.

## Features

* Extracts visible text from webpages.
* Uses TF-IDF vectorization and LDA topic modeling.
* Categorizes content by matching keywords with weighted importance.
* Returns category percentages and main category.

## Setup

### 1. Install dependencies

Make sure you have Python installed, then install required packages via:

```bash
pip install -r requirements.txt
```

### 2. Download NLTK data

Run the `nltk_setup.py` script once to download required NLTK resources:

```bash
python nltk_setup.py
```

This script should include commands like:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## Usage

### Running Locally

Start the Flask app with:

```bash
python app.py
```

Or, using Flask CLI:

```bash
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000
```

### API Endpoint

* **POST** `/analyze`
* Payload example:

```json
{ "url": "https://example.com" }
```

* Response example:

```json
{
  "percentages": {
    "Technology": 45.6,
    "Health": 30.2,
    "Uncategorized": 24.2
  },
  "main_category": "Technology"
}
```

## Deployment on Render

* Connect your repo with `requirements.txt` and `nltk_setup.py`.
* Set build command to:

```bash
pip install -r requirements.txt
python nltk_setup.py
```

* Set start command to:

```bash
python app.py
```

Render will use the `PORT` environment variable automatically.

---

