import pandas as pd
import os
import requests
import tarfile
import io

def load_imdb_data(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading data from local CSVs...")
        return pd.read_csv(train_path), pd.read_csv(test_path)
        
    print("Downloading dataset from Stanford URL... (This takes a few minutes)")
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        print("Download complete. Extracting files...")
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Parsing text files... (This might take a moment)")
        
        def parse_imdb_dir(path):
            data = []
            for label in ["pos", "neg"]:
                dir_path = os.path.join(path, label)
                for fname in os.listdir(dir_path):
                    if fname.endswith(".txt"):
                        with open(os.path.join(dir_path, fname), "r", encoding="utf-8") as f:
                            data.append({"text": f.read(), "label": 1 if label == "pos" else 0})
            return pd.DataFrame(data)
            
        train_df = parse_imdb_dir(os.path.join(data_dir, "aclImdb", "train"))
        test_df = parse_imdb_dir(os.path.join(data_dir, "aclImdb", "test"))
        print("Saving data to local CSVs...")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print("Data loading completely finished!")
        return train_df, test_df
    else:
        raise Exception("Failed to download dataset.")

if __name__ == "__main__":
    train, test = load_imdb_data()
    print(f"Train shape: {train.shape}")
