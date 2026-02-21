
🎭 IMDb Sentiment Analysis Project
Overview
Hi! Welcome to my Sentiment Analysis project. I built this end-to-end Machine Learning pipeline to automatically read movie reviews and figure out if the reviewer actually liked the movie (Positive) or hated it (Negative).

I wanted to build something that goes beyond just training a model in a notebook, so I built out the complete architecture: from downloading the raw data and cleaning the text, to training multiple models, creating visualizations, and finally deploying it all as an interactive web app using Streamlit.

The Dataset
I used the Stanford aclImdb Dataset for this project.

It contains 50,000 movie reviews.

I split it evenly: 25,000 reviews to train the models, and 25,000 brand new reviews to test how accurate they were.

The data is perfectly balanced (50% positive reviews, 50% negative reviews).

How I Built It (The Pipeline)
I organized my code into modular Python scripts inside the src/ folder to keep things clean:

data_loader.py: This script connects to the Stanford server, downloads the 50,000 text files, and converts them into clean Pandas DataFrames (train.csv and test.csv).

preprocessing.py: Raw internet text is messy. I used BeautifulSoup to strip out HTML tags and NLTK to remove stop words and lemmatize the words.

train_ml.py: This is where the actual learning happens. I built Scikit-Learn pipelines and trained two different models to compare their performance: a Naive Bayes model and a Logistic Regression model.

eda.py: I used Matplotlib and WordCloud to visualize the most common words used in good vs. bad reviews.

app.py: Finally, I built a web interface using Streamlit so anyone can type in a custom review and test the models in real-time.

Results
The Logistic Regression model performed the best!

Logistic Regression (TF-IDF): 87.94% Accuracy

Naive Bayes (CountVectorizer): 82.28% Accuracy

Tech Stack Used
Python, Pandas, NumPy

Scikit-Learn (Machine Learning)

NLTK, BeautifulSoup (Natural Language Processing)

Streamlit (Web Deployment)

How to Run This on Your Machine
If you want to test my code on your own computer, here is exactly how to do it:

First, clone this repository and install the required libraries:

bash
git clone https://github.com/AbhiRaj067/imdb_project.git
cd imdb_project
pip install -r requirements.txt
Then, run my scripts in this exact order to process the data and train the models:

bash
python src/data_loader.py
python src/train_ml.py
python src/eda.py
Finally, launch the web app:

bash
streamlit run src/app.py
