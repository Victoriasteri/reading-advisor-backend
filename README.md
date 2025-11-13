# Reading Advisor Backend

A FastAPI-based book recommendation system that uses machine learning models to predict genre, mood, and depth from user queries and recommends matching books.

## Setup

1. **Create a virtual environment (recommended on macOS):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files exist:**

   - `vectorizer.pkl`
   - `model_genre.pkl`
   - `model_mood.pkl`
   - `model_depth.pkl`
   - `books.json`

   If these files don't exist, you'll need to train the models first (see Training section below).

## Running the Server

**Make sure your virtual environment is activated first:**

```bash
source venv/bin/activate
```

Then start the FastAPI server using uvicorn:

```bash
uvicorn server:app --reload
```

The server will start on `http://localhost:8000` by default.

For production, you might want to specify host and port:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. Predict Labels

**POST** `/predict`

Returns the predicted genre, mood, and depth for a given text query.

**Request body:**

```json
{
  "text": "I want something light and cozy to read"
}
```

**Response:**

```json
{
  "genre": "fiction",
  "mood": "cozy",
  "depth": "light"
}
```

### 2. Get Book Recommendation

**POST** `/recommend`

Returns a book recommendation based on the predicted genre, mood, and depth.

**Request body:**

```json
{
  "text": "I want something light and cozy to read"
}
```

**Response:**

```json
{
  "predicted": {
    "genre": "fiction",
    "mood": "cozy",
    "depth": "light"
  },
  "recommendation": {
    "title": "The Little Prince",
    "author": "Antoine de Saint-Exupéry",
    "language": "en",
    "genre": ["fiction"],
    "mood": ["soft", "cozy"],
    "depth": "light",
    "score": 6
  }
}
```

## API Documentation

Once the server is running, you can access:

- **Interactive API docs (Swagger UI):** `http://localhost:8000/docs`
- **Alternative API docs (ReDoc):** `http://localhost:8000/redoc`

## Training Models (Optional)

If you need to retrain the models, use the `train_model.py` script:

```bash
python train_model.py
```

**Note:** This requires a `data/querries.csv` file with columns: `text`, `genre`, `mood`, `depth`.
