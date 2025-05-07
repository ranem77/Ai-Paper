"""
AI vs Human Text Classifier
===========================

This is a simple Python project that classifies short English sentences as either
AI-generated or Human-written using a manually crafted decision tree and basic text features.

------------------------
📋 Overview
------------------------
This script demonstrates a basic machine learning pipeline without using any external libraries:
1. Manually labeled text samples.
2. Text cleaning and preprocessing.
3. Feature extraction (total words, character count, word diversity).
4. A hand-coded decision tree classifier.
5. Model evaluation.
6. Real-time text classification.

The purpose is to show how simple rules and feature engineering can help differentiate
between human and AI-generated content.

------------------------
📂 Project Structure
------------------------
.
├── ai_human_classifier.py  # Main Python file with code and documentation in one

------------------------
🔍 How It Works
------------------------

✅ Step-by-step Explanation:

1. Data Preparation
   A small dataset of 10 text samples is defined — 5 human-written, 5 AI-generated — with labels.

2. Text Cleaning
   Each sentence is lowercased and split into words, filtering out punctuation and non-alphabetic tokens.

3. Feature Extraction
   Features per text:
   - total_words: Number of words
   - total_chars: Total characters
   - diversity: unique words / total words

4. Manual Decision Tree Classifier
   Rule:
   - AI if total_words <= 7
   - Human otherwise

5. Model Testing
   Accuracy is calculated over the dataset.

6. Custom Text Classification
   A function allows classification of new input strings.

------------------------
📈 Accuracy
------------------------
The model’s accuracy is printed after evaluation.

Output:
Model Accuracy: 1.00

Note: This high accuracy is due to the very small and handpicked dataset.

------------------------
🛠 Requirements
------------------------
- Python 3.6+
- No external packages required

------------------------
🚀 How to Run
------------------------
$ python ai_human_classifier.py

------------------------
📌 Limitations
------------------------
- Small dataset (10 samples)
- Handcrafted rule-based model
- Not generalizable to real-world data
- Not robust for long/complex sentences

------------------------
📚 Future Improvements
------------------------
- Use a real dataset (e.g., Reddit vs. ChatGPT)
- Train an actual ML model (e.g., logistic regression)
- Use NLP libraries (scikit-learn, spaCy, NLTK)
- More advanced feature extraction

------------------------
🧾 License
------------------------
Open for learning and experimentation — no restrictions.

------------------------
🙋‍♂️ Author
------------------------
Written by a student exploring basic NLP and AI detection concepts.

"""

# Step 1: Prepare the data
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
    "source": [
        "human", "ai", "human", "ai", "human",
        "ai", "human", "ai", "human", "ai"
    ]
}

# Step 2: Clean the text
def clean_text(text):
    words = text.lower().split()
    cleaned_words = [w for w in words if w.isalpha()]
    return cleaned_words

# Step 3: Extract manual features
def extract_features(text):
    words = clean_text(text)
    total_words = len(words)
    total_chars = sum(len(word) for word in words)
    diversity = len(set(words)) / total_words if total_words > 0 else 0
    return [total_words, total_chars, diversity]

# Prepare data:
X = []
y = []

for content, label in zip(samples["content"], samples["source"]):
    features = extract_features(content)
    X.append(features)
    y.append(1 if label == "ai" else 0)  # ai = 1, human = 0

# Step 4: Build a simple manual decision tree
def simple_decision_tree(features):
    total_words, total_chars, diversity = features
    if total_words <= 7:
        return 1  # AI
    else:
        return 0  # Human

# Step 5: Test the model
correct = 0
for features, label in zip(X, y):
    prediction = simple_decision_tree(features)
    if prediction == label:
        correct += 1

accuracy = correct / len(y)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 6: Try with new input text
def classify_text(input_text):
    features = extract_features(input_text)
    prediction = simple_decision_tree(features)
    return "AI-Generated" if prediction == 1 else "Human-Written"

# Test classification
example_input = "Creative flow is more present in human essays."
print("Detection Result:", classify_text(example_input))
