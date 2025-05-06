import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textstat import flesch_reading_ease
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: NLTK resource setup
nltk.download("punkt")
nltk.download("stopwords")
custom_stopwords = set(stopwords.words("english"))

# Step 2: Custom dataset (new examples)
samples = {
    "content": [
        "Humans write in fragmented ways with emotion.",
        "This message is structured by a neural system.",
        "Natural writing often lacks perfect grammar.",
        "AI-generated outputs are grammatically clean.",
        "Emotion and slang fill most human speech.",
        "AI replies follow a learned statistical pattern.",
        "Mistakes and abbreviations are typical in human typing.",
        "Language model outputs tend to be balanced.",
        "Creative flow is more present in human essays.",
        "Machine-generated text sounds more generic and polished."
    ],
    "source": ["human", "ai", "human", "ai", "human", "ai", "human", "ai", "human", "ai"]
}
df_data = pd.DataFrame(samples)

# Step 3: Text cleaning function
def clean_text(line):
    tokens = word_tokenize(line.lower())
    tokens = [t for t in tokens if t.isalnum() and t not in custom_stopwords]
    return " ".join(tokens)

df_data["clean"] = df_data["content"].apply(clean_text)

# Step 4: Feature engineering
def text_features(line):
    words = word_tokenize(line)
    total_words = len(words)
    total_chars = sum(len(w) for w in words)
    diversity = len(set(words)) / total_words if total_words else 0
    readability = flesch_reading_ease(line)
    return pd.Series([total_words, total_chars, diversity, readability])

df_data[["words", "chars", "diversity", "readability"]] = df_data["content"].apply(text_features)

# Step 5: TF-IDF vectorization
tfidf = TfidfVectorizer(ngram_range=(1, 2))
X_matrix = tfidf.fit_transform(df_data["clean"])
y_labels = df_data["source"].apply(lambda tag: 1 if tag == "ai" else 0)

# Step 6: Data split
X_train, X_test, y_train, y_test = train_test_split(
    X_matrix, y_labels, test_size=0.3, random_state=42, stratify=y_labels
)

# Step 7: Model training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)
print(f"Model Performance (Accuracy): {accuracy_score(y_test, predictions):.2f}")

# Step 8: AI/Human prediction function
def classify_text(input_text):
    cleaned = clean_text(input_text)
    vec_input = tfidf.transform([cleaned])
    result = rf_model.predict(vec_input)[0]
    return "AI-Generated" if result == 1 else "Human-Written"

# Test the function
example_input = "Creative flow is more present in human essays."
print("Detection Result:", classify_text(example_input))
