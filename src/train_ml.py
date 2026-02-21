import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from data_loader import load_imdb_data
from preprocessing import clean_text

def train_baseline_models():
    print("Loading data...")
    train_df, test_df = load_imdb_data()
    print("Preprocessing training data (this takes a few minutes)...")
    train_df['cleaned_text'] = train_df['text'].apply(clean_text)
    test_df['cleaned_text'] = test_df['text'].apply(clean_text)
    
    X_train, y_train = train_df['cleaned_text'], train_df['label']
    X_test, y_test = test_df['cleaned_text'], test_df['label']
    
    print("Training Naive Bayes Model...")
    nb_pipeline = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])
    nb_pipeline.fit(X_train, y_train)
    print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_pipeline.predict(X_test)):.4f}")
    
    print("Training Logistic Regression Model...")
    lr_pipeline = Pipeline([('vect', TfidfVectorizer()), ('clf', LogisticRegression(max_iter=1000))])
    lr_pipeline.fit(X_train, y_train)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pipeline.predict(X_test)):.4f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(nb_pipeline, "models/nb_model.pkl")
    joblib.dump(lr_pipeline, "models/lr_model.pkl")
    print("Models saved successfully!")

if __name__ == "__main__":
    train_baseline_models()
