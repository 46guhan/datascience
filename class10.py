import nltk
import random
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Download NLTK resources (if not already downloaded)
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]



# Shuffle the documents
random.shuffle(documents)

# Preprocessing: Tokenization, Stopwords removal, Lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
print(stop_words)
# Function to preprocess a document
def preprocess(doc):
    words = [lemmatizer.lemmatize(word.lower()) for word in doc if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

# Preprocess each document
preprocessed_documents = [(preprocess(words), category) for words, category in documents]

# Split data into features and labels
X = [doc[0] for doc in preprocessed_documents]
y = [category for _, category in preprocessed_documents]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a classifier (e.g., Linear Support Vector Classifier)
classifier = LinearSVC()
classifier.fit(X_train_tfidf, y_train)

# Predictions
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
 


'''
NLP-Natural Language Processing

depplearning,machince learning,speech recog,cluster,analysis

'''