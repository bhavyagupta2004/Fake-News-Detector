import nltk
nltk.download('stopwords')
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the dataset and perform data preprocessing
# You need to place the dataset loading and preprocessing code here
# Make sure you have a CSV file named 'train.csv' in the same directory as this script

# Load the dataset
news_dataset = pd.read_csv('train.csv')

# Fill missing values with empty strings
news_dataset = news_dataset.fillna('')

# Merge the author name and news title into 'content' column
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Define a stemming function
port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming to the 'content' column
news_dataset['content'] = news_dataset['content'].apply(stemming)

# Separate the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Convert the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Create and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Define a function for fake news detection
def detect_fake_news(text):
    # Preprocess the input text
    text = stemming(text)

    # Convert the text to numerical data using the same vectorizer
    text = vectorizer.transform([text])

    # Make a prediction
    prediction = model.predict(text)

    if prediction[0] == 0:
        return 'The news is Real'
    else:
        return 'The news is Fake'

# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        news_text = request.form['news_text']
        result = detect_fake_news(news_text)
        return render_template('index.html', result=result)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
