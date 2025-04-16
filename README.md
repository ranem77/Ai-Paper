# RoBERTa Text Classifier (Manual PyTorch Implementation)

This project demonstrates how to fine-tune a RoBERTa model using PyTorch without using Hugging Face’s `Trainer` API. The goal is to classify whether a given text is **human-written** or **AI-generated** (like from ChatGPT).

---

## 📌 Features

- Uses `roberta-base` from Hugging Face Transformers.
- Implements custom `Dataset` and `DataLoader`.
- Manual training loop using PyTorch.
- Simple prediction function.
- Includes dummy data; you can replace it with a real dataset like [HC3](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection).

---

## 🧠 Model Overview

The model used is `RobertaForSequenceClassification` with 2 output labels:
- `0`: Human
- `1`: AI (e.g., ChatGPT)

It takes a text input, tokenizes it using `RobertaTokenizer`, and passes it through a fine-tuned classification head.

---

## 🛠️ Requirements

Install all dependencies with:

```bash
pip install torch transformers pandas
```

---

## 📂 Files

- `roberta_text_classifier_manual.py`: Full training and prediction code.
- `README.md`: This documentation.

---

## ▶️ How to Run

1. Save the Python script as `roberta_text_classifier_manual.py`.
2. Run the script:

```bash
python roberta_text_classifier_manual.py
```

---

## 🔎 Code Structure

### 1. Dummy Dataset

Two small samples used to demonstrate the training:

```python
data = {
    'text': [
        "This essay explains economic growth over decades.",  # Human
        "As an AI language model, I believe that the economy is driven by...",  # AI
    ],
    'label': [0, 1]
}
```

### 2. Tokenization

Using `RobertaTokenizer`:

```python
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
```

### 3. Custom Dataset Class

Each sample is tokenized and returned with `input_ids`, `attention_mask`, and `label`.

### 4. DataLoader

Loads data in batches:

```python
loader = DataLoader(dataset, batch_size=2, shuffle=True)
```

### 5. Model and Training

We use `RobertaForSequenceClassification` and train with `AdamW` optimizer and `CrossEntropyLoss`.

### 6. Predict Function

Call:

```python
predict("Your input text here")
```

It returns `"Human"` or `"AI"`.

---

## 📊 Sample Output

```
Epoch 1 Loss: 0.6432
Epoch 2 Loss: 0.3123
Epoch 3 Loss: 0.1834
Prediction: Human
```

---

## 🧪 Tips

- Replace the dummy data with a real labeled dataset for meaningful training.
- Add model saving/loading to avoid retraining each time.
- Evaluate using test accuracy, precision, recall if using a larger dataset.

---

## 📄 License

This project is released for **educational purposes only**. No warranty or performance guarantees are provided. Use at your own risk.

---

## 🙏 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [HC3 Dataset](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)

---

## ✍️ Author

Created by a student developer exploring NLP with RoBERTa and PyTorch 🚀

