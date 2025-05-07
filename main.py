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
