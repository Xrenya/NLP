from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('wordnet')

import spacy

# Stop words
stop_words = set(stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

newsgrous_train = fetch_20newsgroups(subset='train')

categories = newsgrous_train['target_names']

newsgrous_train = fetch_20newsgroups(subset='train',
                                     categories=categories)

def tfidf(text):
    def preproc_nltk(text):
        return ' '.join([lemmatizer.lemmatize(word) for word in 
                        word_tokenize(text.lower())
                        if word not in stop_words])
    vectorizer = TfidfVectorizer(preprocessor=preproc_nltk,
                                 ngram_range=(1, 3),
                                 max_df=0.5,
                                 max_features=1000)
    vectors = vectorizer.fit_transform(text.data)
    return vectors

def spacyfunc(text):
    nlp = spacy.load('en_core_web_sm')
    output = []
    for doc in nlp.pipe(text.data, batch_size=10, n_process=3,
                        disable=['parser', 'ner']):
        output.append(' '.join([token.lemma_ for token in doc
                                if token.lemma_ not in stop_words]))  
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_df=0.5,
        max_features=1000
    )
    vectors = vectorizer.fit_transform(output)
    return vectors

def model(text):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                 max_df=0.5,
                                 max_features=1000)
    vectors = vectorizer.fit_transform(text.data)
    return vectors


def train_model(data, preproc=[tfidf, spacyfunc, model]):
    for prep in preproc:
        vectors = prep(data)
        dense_vectors = vectors.todense()
        X_train, X_test, y_train, y_test = train_test_split(
            dense_vectors, data.target, test_size=0.2,
            random_state=0
        )
        svc = svm.SVC()
        svc.fit(X_train, y_train)
        preds = svc.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print("Accuracy score SVC: ", acc)
        
        f1 = f1_score(y_test, preds, average='weighted')
        print("F1 score SVC: ", f1)
        
        confusion_mat = confusion_matrix(y_test, preds)
        print("Confusion matrix SVC: \n", confusion_mat)
        
        sgd = SGDClassifier()
        sgd.fit(X_train, y_train)
        preds = sgd.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print("Accuracy score SGDClassifier: ", acc)
        
        f1 = f1_score(y_test, preds, average='weighted')
        print("F1 score SGDClassifier: ", f1)
        
        confusion_mat = confusion_matrix(y_test, preds)
        print("Confusion matrix SGDClassifier: \n", confusion_mat)
        

train_model(newsgrous_train)
