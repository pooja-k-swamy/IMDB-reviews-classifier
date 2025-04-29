import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
'''
1. Load your IMDB review dataset (reviews + labels).
2. Clean each review using clean_text.
3. Vectorize cleaned reviews using TfidfVectorizer.
4. Train LogisticRegression on vectorized data.
5. Save the model and vectorizer.
'''

df = pd.read_csv('../sentiment-classifier-serving/model/dataset/IMDB-Dataset.csv')
texts = df['review']
labels = df['sentiment'].map({'positive': 1, 'negative': 0})  # Convert to 0/1

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # remove non-letters
    return text


cleaned_texts = texts.apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(cleaned_texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

model_filename = open("classifier.pkl", "wb")
pickle.dump(model, model_filename)

vectorizer_filename = open("vectorizer.pkl", "wb")
pickle.dump(vectorizer, vectorizer_filename)

print("Training complete. Model and vectorizer saved.")
