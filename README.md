# 🎭 IMDb Movie Review Sentiment Analyzer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://imdbproject-bepzte4hgzqmelgvhps7kd.streamlit.app/)

**Live Demo:** [Try the App Now!](https://imdbproject-bepzte4hgzqmelgvhps7kd.streamlit.app/)

Hi! Welcome to my **IMDb Movie Review Sentiment Analysis** project. This end-to-end Machine Learning pipeline automatically reads movie reviews and predicts if the reviewer liked the movie (🟢 **Positive**) or hated it (🔴 **Negative**).

I built this to demonstrate a complete NLP workflow: from downloading raw data, cleaning messy text, training models, creating visualizations, to deploying a live web app!

## 📊 Dataset
- **Source:** [Stanford Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
- **Size:** 50,000 labeled reviews
- **Train/Test Split:** 25,000 training + 25,000 testing
- **Balance:** Perfectly balanced (50% positive, 50% negative)

## ⚙️ Tech Stack
```text
Python | Pandas | NumPy | Scikit-Learn | NLTK | BeautifulSoup | Streamlit | Matplotlib | Seaborn | WordCloud
🏆 Results
Model	Vectorizer	Accuracy
Logistic Regression	TF-IDF	87.94% 🥇
Naive Bayes	CountVectorizer	82.28% 🥈
🚀 How to Run Locally
If you want to run this project on your own machine, follow these steps:

bash
# 1. Clone the repository
git clone https://github.com/AbhiRaj067/imdb_project.git
cd imdb_project

# 2. Install required libraries
pip install -r requirements.txt

# 3. Download data & train the models (Run in this exact order)
python src/data_loader.py
python src/train_ml.py

# 4. Generate EDA visualizations (Optional)
python src/eda.py

# 5. Launch the Streamlit web app
streamlit run src/app.py
📁 Project Structure
text
imdb_project/
├── README.md                  # Project documentation
├── requirements.txt           # Required Python libraries
├── src/
│   ├── app.py                 # Streamlit web app code
│   ├── data_loader.py         # Connects to Stanford, downloads & parses data
│   ├── preprocessing.py       # Cleans HTML, removes stopwords, lemmatizes text
│   ├── train_ml.py            # Builds pipelines and trains LR & NB models
│   └── eda.py                 # Generates word clouds and distribution plots
├── models/                    # Stores saved .pkl models for quick loading
└── data/                      # Stores CSVs and generated visualization images
