import numpy as np
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1850000
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
stop_words = set(stopwords.words('english'))

def preprocessResponse(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stopwords.words("english")]
    if not tokens:
        return text
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(lemmas)

def themeGrouping(text, n_clusters):
    if text.lower() == 'n/a':
        text = 'Not Available'
    if text != '':
        sentences = sent_tokenize(text)
        preprocessed_sentences = [preprocessResponse(sentence) for sentence in sentences]
        for sentence in preprocessed_sentences:
            if len(word_tokenize(sentence)) <= 1:
                return None  # return None or a default value
            
        tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(tfidf_matrix)

        labels = kmeans.labels_
        themes = {f'{i}': [] for i in range(n_clusters)}

        for i, label in enumerate(labels):
            theme = themes[f'{label}']
            theme.append(sentences[i])
        return themes

def optimalClusters(response):
    if response.lower() == 'n/a':
        return 1
    sentences = sent_tokenize(response)
    max_clusters = len(sentences) - 1
    if len(sentences) < 3:
        return len(sentences)
    preprocessed_sentences = [preprocessResponse(sentence) for sentence in sentences]
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, kmeans.labels_)
        silhouette_scores.append(score)
    
    optimal_clusters = np.argmax(silhouette_scores) + 2
    return optimal_clusters

def refineThemes(results):
    processed_themes = {}
    for theme_id, theme in results.items():
        processed_themes[theme_id] = ' '.join(theme)
    return processed_themes 
   
def main(responses):
    response_list = []
    for response in responses:
        if response.strip()!= '':
            n_clusters = optimalClusters(response)    
            grouped_theme = themeGrouping(response, n_clusters)
            if grouped_theme is not None:
                themes = refineThemes(grouped_theme)
                results = list(themes.values())
                response_list.append(results)
    return response_list