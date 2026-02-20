# Sentiment Analysis Tool
# Project - 2

import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# ----------------------------
# 1. Sample Dataset (No CSV needed)
# ----------------------------
data = {
    "text": [
        "I love this product",
        "This is amazing",
        "Absolutely fantastic experience",
        "Very happy with the service",
        "I hate this",
        "This is terrible",
        "Worst experience ever",
        "Very disappointed",
        "Not good at all",
        "Superb quality",
        "I am so excited",
        "I am very sad",
        "This is not worth it",
        "Highly recommended",
        "Awful customer support"
    ],
    "label": [
        "positive", "positive", "positive", "positive",
        "negative", "negative", "negative", "negative",
        "negative", "positive", "positive", "negative",
        "negative", "positive", "negative"
    ]
}

df = pd.DataFrame(data)


# ----------------------------
# 2. Text Preprocessing Function
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text


df["text"] = df["text"].apply(clean_text)


# ----------------------------
# 3. Convert Text to Features (TF-IDF)
# ----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]


# ----------------------------
# 4. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# ----------------------------
# 5. Train Model (Logistic Regression)
# ----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)


# ----------------------------
# 6. Evaluate Model
# ----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label="positive")

print("\nModel Evaluation")
print("-------------------")
print("Accuracy:", round(accuracy, 2))
print("F1 Score:", round(f1, 2))


# ----------------------------
# 7. CLI for User Input
# ----------------------------
print("\nSentiment Prediction Tool")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter a sentence: ")

    if user_input.lower() == "exit":
        print("Exiting program...")
        break

    cleaned = clean_text(user_input)
    vector_input = vectorizer.transform([cleaned])
    prediction = model.predict(vector_input)

    print("Predicted Sentiment:", prediction[0])
    print()