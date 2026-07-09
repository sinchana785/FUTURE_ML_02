import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

tqdm.pandas()

# Verify all dependencies are present locally
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

def clean_production_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>|http\S+|www\S+|[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in tokens if word not in stop_words])

if __name__ == "__main__":
    print("🚀 Loading raw production dataset...")
    # Since the file is in the same folder, you don't need a long path string anymore!
    df = pd.read_csv("customer_support_tickets.csv")

    print("🗺️ Mapping columns based on Kaggle dataset schema...")
    df = df.rename(columns={
        'Ticket Description': 'ticket_text', 
        'Ticket Type': 'category',        
        'Ticket Priority': 'priority'
    })

    print("🧼 Cleaning text columns across the entire production dataset...")
    df['cleaned_text'] = df['ticket_text'].progress_apply(clean_production_text)

    print("💾 Filtering out missing data and saving a compact copy...")
    df_clean = df[['cleaned_text', 'category', 'priority']].dropna()
    df_clean.to_csv("cleaned_tickets.csv", index=False)

    print(f"✅ Success! Saved to 'cleaned_tickets.csv'.")