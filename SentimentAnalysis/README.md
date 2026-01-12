# Sentiment Analysis Using Python

## 📌 Project Overview
This project performs **Sentiment Analysis** on text data using **Natural Language Processing (NLP)** and **Machine Learning**.  
It classifies user reviews or feedback into **Positive**, **Negative**, or **Neutral** sentiments.

This project is completed as **Task 3 – Sentiment Analysis** under the **Data Science Internship Program**.

---

## 🎯 Objective
- Analyze textual data
- Identify sentiment polarity
- Help businesses understand customer feedback
- Support data-driven decision making

---

## 🛠️ Technologies Used
- Python
- Pandas
- Scikit-learn
- NLTK (Natural Language Toolkit)

---

## 📁 Project Components (Combined in One File)
This README includes:
- Project explanation
- Sample dataset
- Complete Python source code
- Execution steps
- Output description

---

## 📊 Sample Dataset
The dataset contains two columns:
- **text** – user review
- **sentiment** – label (positive / negative / neutral)

```csv
text,sentiment
I love this product,positive
This is very bad,negative
Average experience,neutral
Excellent service,positive
Worst quality ever,negative
🧑‍💻 Complete Python Code
Save the below code as sentiment_analysis.py when executing.

python

import pandas as pd
import nltk
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
from nltk.corpus import stopwords

# Sample dataset
data = {
    'text': [
        'I love this product',
        'This is very bad',
        'Average experience',
        'Excellent service',
        'Worst quality ever'
    ],
    'sentiment': [
        'positive',
        'negative',
        'neutral',
        'positive',
        'negative'
    ]
}

df = pd.DataFrame(data)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = stopwords.words('english')
    return " ".join([word for word in text.split() if word not in stop_words])

df['text'] = df['text'].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# User input testing
while True:
    text = input("\nEnter a sentence (type 'exit' to quit): ")
    if text.lower() == "exit":
        break
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    print("Predicted Sentiment:", prediction[0])
⚙️ Installation Steps
Install required libraries using:


pip install pandas scikit-learn nltk
▶️ How to Run
Open Command Prompt / Terminal

Save the Python code as sentiment_analysis.py

Run the command:

python sentiment_analysis.py
Enter any sentence to predict sentiment.

📈 Output Example

Enter a sentence: I really like this app
Predicted Sentiment: positive

Enter a sentence: very poor service
Predicted Sentiment: negative
🧠 Machine Learning Details
Algorithm: Multinomial Naive Bayes

Vectorization: TF-IDF

Preprocessing:

Lowercasing

Punctuation removal

Stopword removal

✅ Conclusion
This project demonstrates a complete implementation of Sentiment Analysis using Python and Machine Learning.
It effectively classifies text data and can be extended for real-world applications such as customer feedback analysis, review systems, and social media monitoring.

🚀 Future Enhancements
GUI using Tkinter

Web app using Flask/Django

Twitter sentiment analysis

Deep learning models (LSTM / BERT)
