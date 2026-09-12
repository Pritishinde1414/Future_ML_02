import string
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ---------------------------------------------------------
# Step 1: Text Preprocessing (Text Cleaning)
# ---------------------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """Cleans raw text by lowercasing, removing punctuation, 
    eliminating stopwords, and lemmatizing tokens."""
    if not isinstance(text, str):
        return ''

    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words and not word.isdigit()
    ]
    return ' '.join(tokens)

# ---------------------------------------------------------
# Step 2: Load Data (Subject + Description)
# ---------------------------------------------------------
def load_data(filepath='customer_support_tickets.csv'):
    try:
        df = pd.read_csv(filepath)
        
        # Handle missing values in text columns before combining
        df['Ticket Subject'] = df['Ticket Subject'].fillna('')
        df['Ticket Description'] = df['Ticket Description'].fillna('')
        
        # PIPELINE STEP 1: [Ticket Subject] + [Ticket Description]
        df['combined_text'] = df['Ticket Subject'] + " " + df['Ticket Description']
        
        # Map target columns
        df['category'] = df.get('Ticket Type', df.get('Category', pd.Series(dtype='str')))
        df['priority'] = df.get('Ticket Priority', df.get('Priority', pd.Series(dtype='str')))
        
        print(f"Loaded {len(df)} tickets from '{filepath}'.")
        
    except FileNotFoundError:
        print(f"File '{filepath}' not found. Initializing demonstration data...")
        data = {
            'Ticket Subject': [
                'Double Charge on Account', 
                'Login Failed', 
                'SERVER DOWN', 
                'Need old invoices',
                'App Crashing iOS'
            ],
            'Ticket Description': [
                'My credit card was charged twice for the monthly subscription.',
                'Cannot log into my account. Password reset email is never sent.',
                'Critical database connection timeout. All production endpoints down!',
                'Where can I download invoice receipts for the past financial year?',
                'The mobile app crashes immediately upon opening on iOS 17. Urgent fix needed.'
            ],
            'category': ['Billing', 'Account Access', 'Technical Issue', 'Billing', 'Technical Issue'],
            'priority': ['Medium', 'Medium', 'High', 'Low', 'High']
        }
        df = pd.DataFrame(data)
        # PIPELINE STEP 1: [Ticket Subject] + [Ticket Description]
        df['combined_text'] = df['Ticket Subject'] + " " + df['Ticket Description']

    # Drop rows missing target labels
    df = df.dropna(subset=['combined_text', 'category', 'priority'])
    
    # PIPELINE STEP 2: Text Cleaning
    df['cleaned_text'] = df['combined_text'].apply(preprocess_text)
    return df

# ---------------------------------------------------------
# Step 3: Model Pipeline (TF-IDF -> Unigrams/Bigrams -> Logistic Regression)
# ---------------------------------------------------------
def build_classifier_pipeline():
    """Creates the NLP pipeline mapped directly to the architectural diagram."""
    return Pipeline([
        (
            'tfidf',
            # PIPELINE STEP 3 & 4: TF-IDF with Unigrams + Bigrams
            TfidfVectorizer(
                ngram_range=(1, 2), max_features=5000, sublinear_tf=True
            ),
        ),
        (
            'clf',
            # PIPELINE STEP 5: Logistic Regression
            LogisticRegression(
                max_iter=1000, class_weight='balanced', random_state=42
            ),
        ),
    ])

# ---------------------------------------------------------
# Step 4: Execution & Production Simulation
# ---------------------------------------------------------
if __name__ == '__main__':
    # 1. Load and combine Subject + Description
    dataset = load_data('customer_support_tickets.csv')

    # 2. Train Category Model (PIPELINE STEP 6: Category)
    print("\nTraining Category Classifier...")
    X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(
        dataset['cleaned_text'], dataset['category'], test_size=0.2, random_state=42
    )
    category_pipeline = build_classifier_pipeline()
    category_pipeline.fit(X_cat_train, y_cat_train)
    
    cat_preds = category_pipeline.predict(X_cat_test)
    print(f"Category Accuracy: {accuracy_score(y_cat_test, cat_preds) * 100:.2f}%")

    # 3. Train Priority Model (PIPELINE STEP 6: Priority)
    print("Training Priority Classifier...")
    X_pri_train, X_pri_test, y_pri_train, y_pri_test = train_test_split(
        dataset['cleaned_text'], dataset['priority'], test_size=0.2, random_state=42
    )
    priority_pipeline = build_classifier_pipeline()
    priority_pipeline.fit(X_pri_train, y_pri_train)

    pri_preds = priority_pipeline.predict(X_pri_test)
    print(f"Priority Accuracy: {accuracy_score(y_pri_test, pri_preds) * 100:.2f}%")

    # 4. Production Simulation testing the whole flow
    print('\n================ LIVE INFERENCE PIPELINE ================')
    
    # Simulating a new ticket with both Subject and Description
    new_subject = "Payment Gateway 502 Error"
    new_description = "Transactions are failing at checkout with a 502 Bad Gateway response."
    
    # 1. Combine
    combined_input = new_subject + " " + new_description
    
    # 2. Clean
    clean_input = preprocess_text(combined_input)
    
    # 3-6. Vectorize & Predict
    pred_cat = category_pipeline.predict([clean_input])[0]
    pred_pri = priority_pipeline.predict([clean_input])[0]

    print(f"Subject     : {new_subject}")
    print(f"Description : {new_description}")
    print(f"↳ Category  : {pred_cat}")
    print(f"↳ Priority  : {pred_pri}")