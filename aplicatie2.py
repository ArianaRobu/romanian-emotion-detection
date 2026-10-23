import pandas as pd
import re
import unicodedata
import emoji
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

def preprocesare(text):
    text = emoji.demojize(text, language='en')
    text = re.sub(r'http\S+', '', text)
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)\
                      .encode('ascii', 'ignore')\
                      .decode('ascii')
    stopwords = ['in', 'si', 'pe', 'din', 'pentru', 
                 'dintre', 'o', 'un', 'de', 'la', 'cu']
    for cuv in stopwords:
        text = text.replace(' ' + cuv + ' ', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

train['Tweet'] = train['Tweet'].apply(preprocesare)
test['Tweet']  = test['Tweet'].apply(preprocesare)

# Am adaugat max_iter pentru a evita ConvergenceWarning
modele = {
    'LinearSVC':          LinearSVC(max_iter=10000),
    'Naive Bayes':        MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    'SVC': SVC(),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier()
}

param_grids = {
    'LinearSVC': {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__sublinear_tf': [True, False],
    'tfidf__min_df': [1, 2, 3],
    'tfidf__max_df': [0.9, 0.95, 1.0],
    'clf__C': [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
},
    'Logistic Regression': {
        'tfidf__ngram_range': [(1,1), (1,2)],
        'tfidf__sublinear_tf': [True, False],
        'clf__C': [0.1, 1, 10, 100]
    },
    'Naive Bayes': {
        'tfidf__ngram_range': [(1,1), (1,2)],
        'clf__alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
    },
    'SVC': {
        'tfidf__ngram_range': [(1,1), (1,2)],
        'clf__C': [0.1, 1, 10],
        'clf__kernel': ['linear', 'rbf']
    },
    'KNN': {
        'tfidf__ngram_range': [(1,1), (1,2)],
        'clf__n_neighbors': [3, 5, 7, 9, 11, 15]
    },
    'Random Forest': {
        'tfidf__ngram_range': [(1,1), (1,2)],
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 10, 20]
    }
}

for nume, clf in modele.items():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', clf)
    ])
    
    grid = GridSearchCV(pipeline, param_grids[nume], cv=5, scoring='accuracy')
    grid.fit(train['Tweet'], train['Emotion'])
    
    pred = grid.predict(test['Tweet'])
    acc = accuracy_score(test['Emotion'], pred)
    print(f"\n{nume}: best params={grid.best_params_}, test accuracy={acc:.3f}")
    
    if nume == "LinearSVC":
        print("\nRaport de clasificare pentru LinearSVC:")
        print(classification_report(test['Emotion'], pred))

    
    if nume == "LinearSVC":
        best_model = grid.best_estimator_
        vectorizer = best_model.named_steps['tfidf']
        clf_best = best_model.named_steps['clf']
        feature_names = vectorizer.get_feature_names_out()

        for i, emotion in enumerate(clf_best.classes_):
            print(f"\nTop 10 cuvinte pentru Emotia: {emotion}")
            top = clf_best.coef_[i].argsort()[::-1][:10]
            for idx in top:
                print(f"- {feature_names[idx]}")