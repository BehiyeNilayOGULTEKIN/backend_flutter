from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from bs4 import BeautifulSoup
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk import pos_tag, word_tokenize
from collections import defaultdict

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Sample keyword dictionary (expand as needed)
category_keywords = {
    "Technology": {"ai", "machine learning", "algorithm", "data", "software"},
    "Health": {"doctor", "vaccine", "treatment", "hospital", "mental"},
    "Sports": {"football", "tennis", "match", "score", "player"},
    "Politics": {"government", "election", "policy", "president", "minister"},
    "Business": {"market", "finance", "stock", "economy", "company"},
    "Environment": {"climate", "emission", "pollution", "recycle", "sustainability"},
    "Science": {"research", "experiment", "theory", "quantum", "biology"},
    "Art": {"painting", "sculpture", "gallery", "artist", "exhibition"}
}

def collect_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()
        texts = [element.get_text(' ', strip=True) for element in soup.find_all(['h1', 'h2', 'p', 'strong'])]
        return ' '.join(texts)
    except Exception as e:
        print(f"Error accessing URL: {e}")
        return ""

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+|[^a-zA-Z\s]', '', text).lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def filter_meaningful_ngrams(ngrams):
    filtered_ngrams = []
    for phrase in ngrams:
        words = word_tokenize(phrase)
        tags = pos_tag(words)
        if any(tag.startswith("NN") or tag.startswith("JJ") for _, tag in tags):
            filtered_ngrams.append(phrase)
    return filtered_ngrams

def filter_by_pmi(text, threshold=0):
    blob = TextBlob(text)
    ngrams = blob.ngrams(n=2) + blob.ngrams(n=3)
    meaningful_ngrams = [' '.join(ngram) for ngram in ngrams if blob.np_counts.get(' '.join(ngram), 0) > threshold]
    return filter_meaningful_ngrams(meaningful_ngrams)

def analyze_content(text, num_topics=10):
    if not text.strip():
        return None, None, None, None, None
    sentences = nltk.sent_tokenize(text)
    processed_docs = [preprocess_text(sentence) for sentence in sentences if len(sentence.split()) > 3]
    if not processed_docs:
        return None, None, None, None, None
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), min_df=2)
    X = vectorizer.fit_transform(processed_docs)
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=50, learning_method='online', random_state=42)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    topics = [[feature_names[i] for i in topic.argsort()[:-15 - 1:-1]] for topic in lda.components_]
    topic_distribution = lda.transform(X)
    return topics, lda, vectorizer, topic_distribution, processed_docs

def categorize_content(topics, topic_distribution, tfidf_matrix, vectorizer):
    normalized_keywords = {cat: {word.lower() for word in words} for cat, words in category_keywords.items()}
    scores = defaultdict(float)
    feature_names = vectorizer.get_feature_names_out()
    term_importance = tfidf_matrix.sum(axis=0).A1
    word_importance = dict(zip(feature_names, term_importance))
    for topic in topics:
        for word in topic:
            word = word.lower()
            if word in word_importance:
                for category, keywords in normalized_keywords.items():
                    if word in keywords:
                        scores[category] += word_importance[word]
    total = sum(scores.values())
    if total == 0:
        return {"Uncategorized": 100.0}
    percentages = {k: (v / total) * 100 for k, v in scores.items()}
    return dict(sorted(percentages.items(), key=lambda x: x[1], reverse=True))

def analyze_website(url, num_topics=10):
    text = collect_text_from_url(url)
    if not text:
        return {"error": "Failed to retrieve content"}
    meaningful_ngrams = filter_by_pmi(text)
    text += ' ' + ' '.join(meaningful_ngrams)
    topics, lda, vectorizer, topic_distribution, processed_docs = analyze_content(text, num_topics)
    if not topics:
        return {"error": "Text analysis failed"}
    percentages = categorize_content(topics, topic_distribution, vectorizer.transform(processed_docs), vectorizer)
    return {
        "categories": percentages,
        "top_topic_keywords": {f"Topic {i+1}": topic for i, topic in enumerate(topics)}
    }

@app.route('/analyze', methods=['POST'])
def analyze_api():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"error": "Missing URL"}), 400
    result = analyze_website(url)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use 5000 as default if PORT is not set
    app.run(host="0.0.0.0", port=port)
