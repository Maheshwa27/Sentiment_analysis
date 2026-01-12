import pandas as pd
import nltk
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
data = pd.read_csv("dataset.csv")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = stopwords.words('english')
    return " ".join([word for word in text.split() if word not in stop_words])

data['text'] = data['text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['sentiment'], test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# User input test
while True:
    user_text = input("\nEnter a sentence (or type exit): ")
    if user_text.lower() == "exit":
        break
    user_text = clean_text(user_text)
    vector = vectorizer.transform([user_text])
    prediction = model.predict(vector)
    print("Sentiment:", prediction[0])
