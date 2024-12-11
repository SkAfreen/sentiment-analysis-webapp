import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import string
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
import pickle

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv("data.csv")
# data = data.head(500)  # Adjust to test a smaller subset

print(f"Loaded full dataset with {len(data)} entries.")

# Limit dataset to a smaller subset for testing (optional)

print(data.head())

# Text preprocessing function with lemmatization
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize and remove stopwords
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# Apply preprocessing with a progress bar
tqdm.pandas()
data['Cleaned_Text'] = data['text'].progress_apply(clean_text)
print("Preprocessed Data (First 5 Rows):")
print(data[['text', 'Cleaned_Text']].head())

# Split data into training and testing sets
X = data['Cleaned_Text']
y = data['label']  # Ensure the label column corresponds to your dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numeric features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate the model accuracy
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)
print(f"Improved Model Accuracy: {accuracy:.2f}")

# Define parameter grid for Logistic Regression
param_grid = {
    'C': [0.1, 1, 10],  # Regularization strength
    'solver': ['lbfgs', 'liblinear']  # Solver options
}

# Initialize GridSearchCV
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
grid.fit(X_train_vec, y_train)

# Get the best parameters and model
best_params = grid.best_params_
print("Best Parameters from Grid Search:", best_params)

# Use the best model from Grid Search
model = grid.best_estimator_

# Evaluate the tuned model
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)
print(f"Tuned Model Accuracy: {accuracy:.2f}")

# Function to predict sentiment for a given input text
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

# Test the model with various custom inputs
test_inputs = [
    "This movie is absolutely amazing! A must-watch.",
    "I hated this film. It was the worst experience of my life.",
    "The plot was average, but the acting was excellent.",
    "Not bad, but I wouldn’t watch it again.",
    "This movie is the best I have seen this year!"
]

print("\nTesting Custom Inputs:")
for input_text in test_inputs:
    print(f"Input: {input_text}")
    print(f"Predicted Sentiment: {predict_sentiment(input_text)}\n")


# Save the trained model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save the trained vectorizer
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)