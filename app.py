from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import os
import traceback
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from textblob import TextBlob
from nltk import pos_tag
from nltk.tokenize import word_tokenize

nltk.data.path.append('./nltk_data')
# Download resources once
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')


app = Flask(__name__)
CORS(app)

category_keywords = {
    'Technology': [
        'ai', 'artificial intelligence', 'machine learning', 'robotics', 'software', 'hardware',
        'quantum', 'blockchain', 'cybersecurity', 'cloud', 'iot', 'gadget', 'innovation',
        'startup', 'algorithm', 'data', 'programming', 'developer', 'coding', 'technology',
        'smartphone', 'app', 'computing', '5g', 'semiconductor', 'wearable', 'automation',
        'vr', 'ar', 'metaverse', 'digital', 'encryption', 'server', 'firmware', 'opensource',
        'network', 'datacenter', 'processor', 'ai model', 'deep learning', 'neural network',
        'browser', 'database', 'tech company', 'tech news', 'mobile device', 'interface',
        'software update', 'driver', 'kernel', 'UI', 'UX', 'frontend', 'backend', 'framework',
        'edge computing', 'cloud computing', 'data science', 'big data', 'analytics', 'IoT devices',
        'blockchain technology', 'smart city', 'data mining', 'API', 'serverless', 'machine vision',
        'automation tools', 'cyberattack', 'tech startup', 'wireless', '5G technology', 'tech ecosystem',
        'neuralink', 'quantum computing', 'extended reality', 'digital twin', 'autonomous systems',
        'bioinformatics', 'computational linguistics', 'haptic technology', 'brain-computer interface',
        'knowledge representation', 'explainable AI', 'swarm intelligence', 'neuromorphic computing',
        'silicon photonics', 'quantum supremacy'
    ],
    'Business': [
        'market', 'finance', 'investment', 'startup', 'entrepreneur', 'stock', 'revenue',
        'profit', 'loss', 'quarter', 'shareholder', 'IPO', 'merger', 'acquisition',
        'valuation', 'trade', 'economy', 'economic', 'capital', 'banking', 'fund', 'asset',
        'inflation', 'interest rate', 'CEO', 'CFO', 'business', 'deal', 'corporation',
        'enterprise', 'industry', 'forecast', 'strategy', 'dividend', 'bond', 'stock market',
        'index', 'earnings', 'portfolio', 'equity', 'retail', 'logistics', 'supply chain',
        'global market', 'NASDAQ', 'Wall Street', 'product launch', 'sales', 'commerce',
        'invoice', 'marketing', 'brand', 'advertising', 'negotiation', 'startup funding',
        'corporate culture', 'consumer behavior', 'market research', 'business growth',
        'venture capital', 'sustainability', 'economic downturn', 'merger and acquisition', 'digital marketing',
        'fiscal year', 'balance sheet', 'income statement', 'cash flow statement', 'market share',
        'competitive analysis', 'SWOT analysis', 'business plan', 'exit strategy', 'due diligence',
        'shareholder value', 'customer acquisition cost', 'lifetime value', 'churn rate',
        'gross margin', 'operating margin', 'net profit margin', 'return on investment', 'return on equity',
        'debt-to-equity ratio', 'price-to-earnings ratio', 'earnings per share', 'market capitalization',
        'venture capital funding', 'private equity', 'angel investor', 'seed funding', 'series A funding',
        'series B funding', 'burn rate', 'runway', 'bootstrapping', 'crowdfunding', 'acqui-hire',
        'initial public offering', 'secondary offering', 'dilution', 'market penetration', 'market segmentation',
        'target market', 'marketing mix', 'product positioning', 'value proposition', 'competitive advantage',
        'economies of scale', 'supply and demand', 'fiscal policy', 'monetary policy', 'GDP growth',
        'unemployment rate', 'consumer price index', 'producer price index', 'trade deficit', 'trade surplus'
    ],
    'Sports': [
        'match', 'tournament', 'team', 'player', 'score', 'goal', 'coach', 'league', 'season',
        'championship', 'cup', 'soccer', 'basketball', 'tennis', 'athlete', 'medal', 'record',
        'olympics', 'game', 'referee', 'injury', 'win', 'lose', 'draw', 'pitch', 'field',
        'court', 'fans', 'supporters', 'training', 'draft', 'transfer', 'competition',
        'event', 'stadium', 'matchday', 'MVP', 'sportsman', 'sportswoman', 'FIFA', 'NBA',
        'NFL', 'scoreboard', 'freekick', 'penalty', 'run', 'goalkeeper', 'offense', 'defense',
        'manager', 'fixture', 'replay', 'highlight', 'champion', 'athletics', 'rugby', 'swimming',
        'formula 1', 'boxing', 'wrestling', 'golf', 'teamwork', 'referee', 'sports psychology',
        'fitness training', 'game strategy', 'sports nutrition', 'track and field',
        'home run', 'touchdown', 'slam dunk', 'ace', 'birdie', 'par', 'bogey', 'hole-in-one',
        'knockout', 'roundhouse kick', 'submission', 'grand slam', 'world series', 'super bowl',
        'stanley cup', 'premier league', 'la liga', 'serie a', 'bundesliga', 'tour de france',
        'wimbledon', 'us open', 'masters tournament', 'olympic games', 'paralympic games',
        'x games', 'extreme sports', 'motorsports', 'nascar', 'rally racing', 'triathlon',
        'marathon', 'ironman', 'decathlon', 'heptathlon', 'steeplechase', 'hurdles', 'javelin',
        'discus', 'hammer throw', 'shot put', 'pole vault', 'high jump', 'long jump',
        'triple jump', 'relay race', 'sprint', 'middle distance', 'long distance', 'cross country',
        'speed skating', 'figure skating', 'ice hockey', 'curling', 'bobsled', 'luge', 'skeleton',
        'ski jumping', 'snowboarding', 'freestyle skiing', 'alpine skiing', 'biathlon', 'nordic skiing'
    ],
    'Health': [
        'health', 'wellness', 'disease', 'illness', 'infection', 'hospital', 'clinic', 'doctor',
        'nurse', 'vaccine', 'medicine', 'treatment', 'diagnosis', 'therapy', 'mental health',
        'nutrition', 'fitness', 'exercise', 'diet', 'obesity', 'virus', 'pandemic', 'epidemic',
        'covid', 'surgery', 'pharmaceutical', 'symptom', 'immune', 'pain', 'chronic',
        'cancer', 'diabetes', 'heart disease', 'blood pressure', 'prescription', 'public health',
        'hygiene', 'recovery', 'checkup', 'injection', 'infection', 'antibiotic', 'healthcare',
        'ER', 'ICU', 'vital signs', 'anxiety', 'depression', 'meditation', 'yoga', 'rest',
        'mental illness', 'healthy lifestyle', 'physical therapy', 'well-being', 'immunity',
        'sleep hygiene', 'rehabilitation', 'health insurance', 'chronic illness', 'health checkup',
        'oncology', 'cardiology', 'neurology', 'dermatology', 'ophthalmology', 'otolaryngology',
        'urology', 'gynecology', 'pediatrics', 'geriatrics', 'endocrinology', 'gastroenterology',
        'pulmonology', 'nephrology', 'hematology', 'immunology', 'rheumatology', 'allergy',
        'infectious disease', 'emergency medicine', 'family medicine', 'internal medicine',
        'preventive medicine', 'alternative medicine', 'complementary medicine', 'integrative medicine',
        'traditional medicine', 'holistic health', 'wellness coaching', 'health promotion',
        'disease prevention', 'health education', 'health literacy', 'patient care', 'medical research',
        'clinical trial', 'evidence-based medicine', 'personalized medicine', 'precision medicine',
        'regenerative medicine', 'telemedicine', 'e-health', 'm-health', 'public health policy',
        'global health', 'community health', 'occupational health', 'environmental health',
        'nutritional science', 'exercise physiology', 'sports medicine', 'rehabilitation medicine'
    ],
    'Education': [
        'school', 'university', 'college', 'curriculum', 'teacher', 'student', 'exam', 'grade',
        'degree', 'education', 'learning', 'classroom', 'syllabus', 'tuition', 'lecture',
        'homework', 'assignment', 'textbook', 'e-learning', 'online class', 'distance learning',
        'scholarship', 'academic', 'GPA', 'enrollment', 'course', 'graduation', 'research',
        'campus', 'professor', 'seminar', 'certificate', 'MOOC', 'educator', 'principal',
        'school board', 'lab', 'quiz', 'essay', 'literacy', 'dropout', 'kindergarten',
        'high school', 'SAT', 'ACT', 'PhD', 'master’s', 'bachelor’s', 'adult education',
        'online degree', 'study abroad', 'peer learning', 'STEM education', 'academic research',
        'learning platform', 'curriculum design', 'digital classroom', 'pedagogy', 'andragogy',
        'didactics', 'epistemology', 'cognitive development', 'learning styles', 'multiple intelligences',
        'constructivism', 'behaviorism', 'cognitivism', 'connectivism', 'universal design for learning',
        'differentiated instruction', 'inquiry-based learning', 'project-based learning',
        'experiential learning', 'cooperative learning', 'collaborative learning',
        'problem-based learning', 'place-based education', 'service-learning', 'social-emotional learning',
        'culturally responsive teaching', 'trauma-informed teaching', 'flipped classroom',
        'blended learning', 'personalized learning', 'adaptive learning', 'gamification',
        'educational technology', 'instructional design', 'assessment', 'formative assessment',
        'summative assessment', 'standardized testing', 'portfolio assessment', 'rubrics',
        'learning outcomes', 'competency-based education', 'lifelong learning', 'continuing education',
        'vocational education', 'technical education', 'special education', 'inclusive education',
        'bilingual education', 'multilingual education', 'early childhood education',
        'elementary education', 'secondary education', 'higher education', 'postsecondary education',
        'graduate education', 'postgraduate education', 'distance education'
    ],
    'Politics': [
        'election', 'vote', 'government', 'president', 'minister', 'parliament', 'policy',
        'political', 'democracy', 'republic', 'campaign', 'debate', 'constitution', 'legislation',
        'bill', 'senate', 'congress', 'party', 'diplomacy', 'governor', 'ambassador',
        'coalition', 'reform', 'law', 'justice', 'supreme court', 'bureaucracy', 'diplomatic',
        'cabinet', 'prime minister', 'mayor', 'foreign affairs', 'national security', 'agenda',
        'mandate', 'scandal', 'voter', 'referendum', 'authority', 'executive', 'legislative',
        'politician', 'ministerial', 'summit', 'opposition', 'policy making', 'propaganda',
        'political party', 'state department', 'political ideology', 'lobbying', 'public opinion',
        'political system', 'electoral system', 'political process', 'political participation',
        'political culture', 'political discourse', 'political communication', 'political rhetoric',
        'political philosophy', 'political theory', 'political science', 'comparative politics',
        'international relations', 'geopolitics', 'public administration', 'public policy',
        'civic engagement', 'civil society', 'social movement', 'political activism',
        'grassroots movement', 'political organization', 'non-governmental organization',
        'interest group', 'pressure group', 'think tank', 'political analysis', 'political commentary',
        'political satire', 'political drama', 'political thriller', 'political intrigue',
        'political power', 'political influence', 'political leadership', 'political elite',
        'political dynasty', 'political machine', 'political patronage', 'political corruption',
        'political accountability', 'political transparency', 'political reform', 'political change',
        'political transition', 'political stability', 'political instability', 'political crisis',
        'political conflict', 'political violence', 'political repression', 'political oppression',
        'political liberation', 'political revolution', 'political independence', 'political sovereignty',
        'political self-determination', 'political union', 'political integration', 'political fragmentation'
    ],
    'Science': [
        'research', 'experiment', 'theory', 'scientist', 'lab', 'data', 'hypothesis', 'chemistry',
        'biology', 'physics', 'astronomy', 'genetics', 'discovery', 'climate', 'space', 'nasa',
        'microscope', 'analysis', 'observation', 'geology', 'ecology', 'element', 'equation',
        'molecule', 'atom', 'nucleus', 'energy', 'gravity', 'invention', 'calculation',
        'scientific method', 'carbon', 'innovation', 'telescope', 'scientific journal',
        'experiment result', 'DNA', 'RNA', 'reaction', 'test tube', 'publication', 'space exploration',
        'nanotechnology', 'robotics', 'artificial intelligence', 'biotechnology', 'climate change',
        'greenhouse gases', 'quantum mechanics', 'thermodynamics', 'biomaterials', 'neuroscience',
        'particle physics', 'cosmology', 'astrophysics', 'organic chemistry', 'inorganic chemistry',
        'analytical chemistry', 'physical chemistry', 'biochemistry', 'molecular biology',
        'cell biology', 'microbiology', 'immunology', 'physiology', 'anatomy', 'zoology',
        'botany', 'marine biology', 'evolutionary biology', 'genomics', 'proteomics',
        'bioinformatics', 'systems biology', 'synthetic biology', 'biophysics', 'biogeochemistry',
        'climatology', 'meteorology', 'oceanography', 'seismology', 'volcanology',
        'paleontology', 'anthropology', 'archaeology', 'sociology', 'psychology',
        'cognitive science', 'linguistics', 'computer science', 'information theory',
        'systems theory', 'complexity theory', 'chaos theory', 'network science',
        'materials science', 'solid-state physics', 'condensed matter physics',
        'nuclear physics', 'plasma physics', 'optics', 'acoustics', 'mechanics',
        'fluid dynamics', 'thermodynamics', 'electromagnetism', 'relativity',
        'quantum field theory', 'string theory', 'unified field theory'
    ],
    'Entertainment': [
        'movie', 'film', 'actor', 'actress', 'cinema', 'tv', 'show', 'series', 'music', 'album',
        'song', 'concert', 'celebrity', 'award', 'theater', 'drama', 'comedy', 'performance',
        'festival', 'streaming', 'netflix', 'hollywood', 'bollywood', 'director', 'trailer',
        'ticket', 'episode', 'season', 'box office', 'celebrity news', 'oscars', 'emmys',
        'documentary', 'soundtrack', 'dance', 'reality show', 'broadway', 'animation', 'screenplay',
        'musical', 'TV series', 'celebrity gossip', 'casting', 'red carpet', 'streaming platform',
        'film production', 'indie film', 'acting', 'director’s cut', 'blockbuster', 'audition',
        'genre', 'narrative', 'plot', 'character', 'theme', 'style', 'tone', 'mood',
        'setting', 'mise-en-scène', 'cinematography', 'editing', 'sound design', 'visual effects',
        'special effects', 'makeup', 'costume design', 'production design', 'art direction',
        'screenwriter', 'screenplay adaptation', 'dialogue', 'monologue', 'voiceover',
        'subtitles', 'dubbing', 'premiere', 'release', 'distribution', 'box office success',
        'critical acclaim', 'audience reception', 'cult film', 'indie film festival',
        'film critic', 'film review', 'film analysis', 'film theory', 'film history',
        'film movement', 'film genre', 'film style', 'film technique', 'film aesthetic',
        'film culture', 'film industry', 'film market', 'film financing', 'film production company',
        'film studio', 'film distributor', 'film exhibitor', 'film festival', 'film award',
        'film director', 'film producer', 'film actor', 'film actress', 'film crew',
        'film composer', 'film editor', 'film cinematographer', 'film production designer',
        'film costume designer', 'film makeup artist', 'film visual effects supervisor',
        'film sound designer', 'film screenwriter', 'film critic', 'film historian',
        'film theorist', 'film scholar', 'film buff', 'cinephile', 'moviegoer'
    ],
    'Environment': [
        'climate', 'global warming', 'pollution', 'emission', 'carbon footprint', 'ecology',
        'biodiversity', 'sustainability', 'recycling', 'greenhouse gas', 'conservation', 'wildlife',
        'natural resource', 'renewable', 'solar', 'wind energy', 'deforestation', 'fossil fuel',
        'ocean', 'marine', 'forest', 'tree', 'eco-friendly', 'sustainable', 'plastic waste',
        'environmental', 'earth', 'climate change', 'carbon neutral', 'green energy', 'air quality',
        'water pollution', 'soil', 'ozone layer', 'environmentalist', 'carbon tax', 'zero waste',
        'environmental policy', 'green technology', 'sustainable development', 'forest conservation',
        'alternative energy', 'clean energy', 'wildlife preservation', 'ecosystem', 'habitat',
        'species', 'extinction', 'endangered species', 'invasive species', 'conservation biology',
        'environmental science', 'environmental studies', 'environmental engineering',
        'environmental law', 'environmental ethics', 'environmental justice',
        'environmental impact assessment', 'ecological footprint', 'carbon sequestration',
        'renewable energy sources', 'sustainable agriculture', 'organic farming',
        'permaculture', 'agroecology', 'sustainable forestry', 'sustainable fishing',
        'sustainable tourism', 'ecotourism', 'green building', 'sustainable architecture',
        'energy efficiency', 'resource management', 'waste management', 'water conservation',
        'air pollution control', 'water treatment', 'wastewater treatment', 'solid waste management',
        'hazardous waste management', 'noise pollution', 'light pollution',
        'electromagnetic pollution', 'radioactive contamination', 'land degradation',
        'desertification', 'drought', 'flood', 'sea level rise', 'ocean acidification',
        'coral bleaching', 'biodegradable', 'compostable', 'recyclable materials',
        'circular economy', 'life cycle assessment', 'environmental management systems',
        'corporate sustainability', 'sustainable consumption', 'sustainable lifestyle'
    ],
    'Art': [
        'art', 'painting', 'sculpture', 'gallery', 'exhibition', 'artist', 'drawing', 'illustration',
        'museum', 'design', 'creative', 'canvas', 'portrait', 'modern art', 'abstract', 'installation',
        'fine art', 'aesthetics', 'artwork', 'curator', 'visual art', 'expression', 'color theory',
        'art history', 'ink', 'sketch', 'brush', 'composition', 'perspective', 'contemporary',
        'masterpiece', 'artistic', 'medium', 'craft', 'ceramics', 'printmaking', 'graffiti',
        'performance art', 'art fair', 'still life', 'self-portrait', 'collage',
        'street art', 'calligraphy', 'art installation', 'mixed media', 'sculptor',
        'fine arts degree', 'art studio', 'fine arts exhibition', 'surrealism', 'expressionism',
        'art movement', 'art style', 'art technique', 'art criticism', 'art theory',
        'art appreciation', 'art collection', 'art market', 'art auction', 'art dealer',
        'art collector', 'art patron', 'art commission', 'art prize', 'art residency',
        'art school', 'art student', 'art teacher', 'art professor', 'art workshop',
        'art therapy', 'community art', 'public art', 'mural', 'environmental art',
        'eco-art', 'feminist art', 'political art', 'social commentary', 'cultural identity',
        'heritage', 'tradition', 'avant-garde', 'experimental', 'interdisciplinary',
        'transmedia', 'conceptual art', 'minimalism', 'postmodernism', 'land art',
        'body art', 'sound art', 'light art', 'kinetic art', 'op art', 'pop art',
        'neo-expressionism', 'outsider art', 'visionary art', 'naive art', 'folk art',
        'tribal art', 'indigenous art', 'ancient art', 'classical art', 'baroque art',
        'rococo art', 'gothic art', 'renaissance art', 'neoclassical art', 'romantic art',
        'realist art', 'impressionist art', 'post-impressionist art', 'symbolist art',
        'art nouveau', 'fauvism', 'cubism', 'futurism', 'dadaism', 'surrealism',
        'expressionism', 'abstract expressionism', 'color field painting', 'hard-edge painting',
        'lyrical abstraction', 'neo-dada', 'fluxus', 'happening', 'performance art',
        'video art', 'digital art', 'new media art', 'interactive art', 'net art',
        'sound art', 'light art', 'installation art', 'site-specific art', 'public art',
        'land art', 'earth art', 'environmental art', 'eco-art', 'feminist art',
        'political art', 'social commentary art', 'cultural identity art',
        'heritage art', 'tradition art', 'avant-garde art', 'experimental art',
        'interdisciplinary art', 'transmedia art'
    ]
}

def collect_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()
        texts = [element.get_text(' ', strip=True) for element in soup.find_all(['h1', 'h2','h3','h4','h5','h6', 'p', 'strong'])]
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

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        text = " ".join(p.get_text() for p in soup.find_all(["p", "h1", "h2"]))

        processed_docs = preprocess_text(text)
        tfidf = vectorizer.transform(processed_docs)
        topic_distribution = lda.transform(tfidf)
        percentages = categorize_content(topics, topic_distribution, tfidf, vectorizer)
        main_category = max(percentages.items(), key=lambda x: x[1])

        return jsonify({
            "percentages": percentages,
            "main_category": main_category[0]
        })

    except Exception as e:
        print("ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
