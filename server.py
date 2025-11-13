import pickle
import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

# ---------- Load ML models ----------
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model_genre.pkl", "rb") as f:
    model_genre = pickle.load(f)

with open("model_mood.pkl", "rb") as f:
    model_mood = pickle.load(f)

with open("model_depth.pkl", "rb") as f:
    model_depth = pickle.load(f)

# ---------- Load books ----------
BOOKS_PATH = Path(__file__).parent / "books.json"

with open(BOOKS_PATH, "r", encoding="utf-8") as f:
    BOOKS = json.load(f)


class Query(BaseModel):
    text: str


app = FastAPI()


def predict_labels(text: str):
    """Use ML models to predict genre, mood, depth for a given text."""
    X = vectorizer.transform([text])
    genre = model_genre.predict(X)[0]
    mood = model_mood.predict(X)[0]
    depth = model_depth.predict(X)[0]
    return genre, mood, depth


def score_book(book: dict, genre: str, mood: str, depth: str) -> int:
    """
    Simple scoring:
      +3 if genre matches
      +2 if depth matches
      +1 if mood is contained in book["mood"]
    """
    score = 0

    if genre in book.get("genre", []):
        score += 3

    if depth == book.get("depth"):
        score += 2

    if mood in book.get("mood", []):
        score += 1

    return score


@app.post("/predict")
def predict(query: Query):
    """Return just the raw ML predictions."""
    genre, mood, depth = predict_labels(query.text)
    return {
        "genre": genre,
        "mood": mood,
        "depth": depth
    }


@app.post("/recommend")
def recommend(query: Query):
    """
    Use ML predictions + books.json to recommend the best matching book.
    """
    genre, mood, depth = predict_labels(query.text)

    best_book = None
    best_score = -1

    for book in BOOKS:
        s = score_book(book, genre, mood, depth)
        if s > best_score:
            best_score = s
            best_book = book

    if best_book is None:
        return {
            "message": "No suitable book found",
            "predicted": {
                "genre": genre,
                "mood": mood,
                "depth": depth
            }
        }

    return {
        "predicted": {
            "genre": genre,
            "mood": mood,
            "depth": depth
        },
        "recommendation": {
            "title": best_book["title"],
            "author": best_book["author"],
            "language": best_book["language"],
            "genre": best_book["genre"],
            "mood": best_book["mood"],
            "depth": best_book["depth"],
            "score": best_score
        }
    }
