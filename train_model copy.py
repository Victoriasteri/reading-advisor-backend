import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("data/querries.csv")

texts = df["text"].astype(str)
y_genre = df["genre"].astype(str)
y_mood = df["mood"].astype(str)
y_depth = df["depth"].astype(str)

# Split indices first, then use them to split all data consistently
train_indices, test_indices = train_test_split(
    df.index, test_size=0.2, random_state=42
)

X_train_text = texts.loc[train_indices]
X_test_text = texts.loc[test_indices]
y_train_genre = y_genre.loc[train_indices]
y_test_genre = y_genre.loc[test_indices]
y_train_mood = y_mood.loc[train_indices]
y_test_mood = y_mood.loc[test_indices]
y_train_depth = y_depth.loc[train_indices]
y_test_depth = y_depth.loc[test_indices]

# Vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ---------- Train Genre ----------
genre_model = LogisticRegression(max_iter=1000)
genre_model.fit(X_train, y_train_genre)
genre_acc = genre_model.score(X_test, y_test_genre)
print("Genre accuracy:", genre_acc)

# ---------- Train Mood ----------
mood_model = LogisticRegression(max_iter=1000)
mood_model.fit(X_train, y_train_mood)
mood_acc = mood_model.score(X_test, y_test_mood)
print("Mood accuracy:", mood_acc)

# ---------- Train Depth ----------
depth_model = LogisticRegression(max_iter=1000)
depth_model.fit(X_train, y_train_depth)
depth_acc = depth_model.score(X_test, y_test_depth)
print("Depth accuracy:", depth_acc)

# Save vectorizer + models
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model_genre.pkl", "wb") as f:
    pickle.dump(genre_model, f)

with open("model_mood.pkl", "wb") as f:
    pickle.dump(mood_model, f)

with open("model_depth.pkl", "wb") as f:
    pickle.dump(depth_model, f)

print("All models saved!")
