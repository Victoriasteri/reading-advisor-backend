import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load data
df = pd.read_csv("data/querries.csv")

print(f"Total training samples: {len(df)}")
print("\nClass distributions:")
print("Genre:", df["genre"].value_counts().to_dict())
print("Mood:", df["mood"].value_counts().to_dict())
print("Depth:", df["depth"].value_counts().to_dict())
print("\n" + "="*50 + "\n")

texts = df["text"].astype(str)
y_genre = df["genre"].astype(str)
y_mood = df["mood"].astype(str)
y_depth = df["depth"].astype(str)

# Split indices first, then use them to split all data consistently
# Stratify by genre to maintain class distribution
train_indices, test_indices = train_test_split(
    df.index, test_size=0.2, random_state=42, stratify=y_genre
)

X_train_text = texts.loc[train_indices]
X_test_text = texts.loc[test_indices]
y_train_genre = y_genre.loc[train_indices]
y_test_genre = y_genre.loc[test_indices]
y_train_mood = y_mood.loc[train_indices]
y_test_mood = y_mood.loc[test_indices]
y_train_depth = y_depth.loc[train_indices]
y_test_depth = y_depth.loc[test_indices]

# Improved Vectorizer with better parameters
# - ngram_range=(1,2): captures both single words and word pairs
# - min_df=2: ignore terms that appear in less than 2 documents
# - max_df=0.95: ignore terms that appear in more than 95% of documents
# - lowercase and strip_accents for better normalization
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  # Use unigrams and bigrams
    min_df=2,  # Minimum document frequency
    max_df=0.95,  # Maximum document frequency
    lowercase=True,
    strip_accents='unicode',
    max_features=5000  # Limit features to prevent overfitting
)

print("Fitting vectorizer...")
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
print(f"Feature matrix shape: {X_train.shape}\n")

# Function to train and evaluate a model
def train_and_evaluate(X_train, y_train, X_test, y_test, task_name, class_weight='balanced'):
    """Train a model and print detailed evaluation metrics."""
    print(f"Training {task_name} model...")
    
    # Use balanced class weights to handle imbalanced data
    model = LogisticRegression(
        max_iter=2000,
        class_weight=class_weight,  # Automatically balance classes
        C=1.0,  # Regularization strength
        solver='lbfgs'  # Good for small datasets, automatically uses multinomial for multi-class
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    
    print(f"\n{task_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\n" + "-"*50 + "\n")
    
    return model, accuracy

# ---------- Train Genre Model ----------
genre_model, genre_acc = train_and_evaluate(
    X_train, y_train_genre, X_test, y_test_genre, "Genre"
)

# ---------- Train Mood Model ----------
mood_model, mood_acc = train_and_evaluate(
    X_train, y_train_mood, X_test, y_test_mood, "Mood"
)

# ---------- Train Depth Model ----------
depth_model, depth_acc = train_and_evaluate(
    X_train, y_train_depth, X_test, y_test_depth, "Depth"
)

# Save vectorizer + models
print("Saving models...")
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model_genre.pkl", "wb") as f:
    pickle.dump(genre_model, f)

with open("model_mood.pkl", "wb") as f:
    pickle.dump(mood_model, f)

with open("model_depth.pkl", "wb") as f:
    pickle.dump(depth_model, f)

print("\n" + "="*50)
print("All models saved successfully!")
print("="*50)
print(f"\nFinal Accuracies:")
print(f"  Genre: {genre_acc:.4f}")
print(f"  Mood:  {mood_acc:.4f}")
print(f"  Depth: {depth_acc:.4f}")
