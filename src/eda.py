import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from data_loader import load_imdb_data
from preprocessing import clean_text

def run_eda():
    print("Loading data for EDA...")
    train_df, _ = load_imdb_data()
    
    print("Creating class distribution chart...")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=train_df)
    plt.title('Class Distribution')
    plt.savefig("data/class_distribution.png")
    plt.close()

    print("Cleaning text for Word Clouds...")
    subset = train_df.sample(2000, random_state=42)
    subset['cleaned_text'] = subset['text'].apply(clean_text)
    
    print("Generating Word Clouds...")
    pos_text = " ".join(subset[subset['label'] == 1]['cleaned_text'])
    neg_text = " ".join(subset[subset['label'] == 0]['cleaned_text'])
    
    WordCloud(width=800, height=400, background_color='white').generate(pos_text).to_file("data/pos_wordcloud.png")
    WordCloud(width=800, height=400, background_color='white').generate(neg_text).to_file("data/neg_wordcloud.png")
    print("Visualizations saved successfully!")

if __name__ == "__main__":
    run_eda()
