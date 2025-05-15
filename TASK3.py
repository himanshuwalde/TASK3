# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Load Twitter dataset
df = pd.read_csv(r"C:\Users\VICTUS\Downloads\Twitter_Data.csv", nrows=10000)
print(df.head())

# Clean data - Twitter uses -1 (negative), 0 (neutral), 1 (positive)
df = df.rename(columns={'clean_text': 'Text', 'category': 'Score'})  # Adjust column names if needed
df = df.dropna()  # Remove empty rows

# Convert numeric scores to labels
df['Sentiment'] = df['Score'].map({
    -1: 'negative',
    0: 'neutral',
    1: 'positive'
})

print("\nSentiment Distribution:")
print(df['Sentiment'].value_counts())

# Visualize
plt.figure(figsize=(8,5))
df['Sentiment'].value_counts().plot(kind='bar', color=['green', 'gray', 'red'])
plt.title("Twitter Sentiment Distribution")
plt.show()

# Prepare data for ML
X = df['Text']
y = df['Sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numbers
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Test model
predictions = model.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, predictions))

# Confusion matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['negative', 'neutral', 'positive'], 
            yticklabels=['negative', 'neutral', 'positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()