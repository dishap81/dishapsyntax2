📊 Sentiment Analysis Tool
📌 Project Overview
This project is a simple Machine Learning based Sentiment Analysis Tool that classifies text as Positive or Negative.
It uses text preprocessing, TF-IDF feature extraction, and Logistic Regression for classification.
🎯 Objective
To build a model that can analyze user input sentences and predict whether the sentiment is positive or negative.
🛠 Technologies Used
Python
Pandas
Scikit-learn
TF-IDF Vectorizer
Logistic Regression
⚙️ Methodology
Data Preparation
Used a small labeled dataset of positive and negative sentences.
Text Preprocessing
Converted text to lowercase
Removed punctuation
Removed numbers
Feature Extraction
Converted text into numerical features using TF-IDF Vectorizer.
Model Training
Used Logistic Regression for classification.
Model Evaluation
Evaluated using:
Accuracy Score
F1 Score
Prediction Interface
Implemented a CLI where users can input sentences.
The model predicts and displays the sentiment.
▶️ How to Run
Install required libraries:
Copy code

pip install pandas scikit-learn
Run the program:
Copy code

python sentiment_analysis.py
Enter any sentence to get predicted sentiment.
📈 Output
The program displays:
Model Accuracy
F1 Score
Predicted Sentiment for user input
✅ Conclusion
This project demonstrates how Natural Language Processing (NLP) techniques and Machine Learning algorithms can be used to perform sentiment classification on text data.# dishapsyntax2
