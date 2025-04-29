from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from model.modelTrain import clean_text

# Load model and vectorizer
model = pickle.load(open('../sentiment-classifier-serving/model/classifier.pkl', 'rb'))
vectorizer = pickle.load(open('../sentiment-classifier-serving/model/vectorizer.pkl', 'rb'))

app = FastAPI()


class Review(BaseModel):
    text: str


@app.post('/predictSentiment')
def predict_sentiment(review: Review):
    cleaned = clean_text(review.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return {"sentiment": "positive" if prediction == 1 else "negative"}

