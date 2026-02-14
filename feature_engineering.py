"""
Feature Engineering for Escalation Risk Classifier
Extracts text-based and metadata features to improve model performance
"""

import pandas as pd
import numpy as np

def extract_text_features(text):
    """
    Extract engineered features from complaint text
    
    Args:
        text (str): Customer complaint text
        
    Returns:
        dict: Dictionary of features
    """
    if pd.isna(text) or not isinstance(text, str):
        text = ""
    
    features = {}
    
    # Basic text statistics
    features['char_length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    
    # Capitalization features (anger indicator)
    features['caps_ratio'] = sum(c.isupper() for c in text) / len(text) if len(text) > 0 else 0
    features['all_caps_words'] = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
    
    # Punctuation features (emotion indicators)
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['multiple_exclamation'] = text.count('!!') + text.count('!!!') + text.count('!!!!')
    
    # Urgency indicators
    urgency_words = ['urgent', 'immediately', 'asap', 'emergency', 'critical', 
                     'right now', 'right away', 'as soon as possible']
    text_lower = text.lower()
    features['urgency_score'] = sum(1 for word in urgency_words if word in text_lower)
    
    # Anger/frustration indicators
    anger_words = ['furious', 'angry', 'outraged', 'disgusted', 'horrible', 
                   'worst', 'terrible', 'unacceptable', 'ridiculous', 'pathetic']
    features['anger_score'] = sum(1 for word in anger_words if word in text_lower)
    
    # Legal threat indicators (VERY important for escalation)
    legal_words = ['lawyer', 'attorney', 'sue', 'lawsuit', 'legal action', 
                   'court', 'judge', 'litigation', 'class action']
    features['legal_threat'] = sum(1 for word in legal_words if word in text_lower)
    
    # Fraud/crime indicators
    fraud_words = ['fraud', 'scam', 'theft', 'steal', 'stolen', 'rob', 
                   'criminal', 'illegal', 'crime']
    features['fraud_score'] = sum(1 for word in fraud_words if word in text_lower)
    
    # Escalation action indicators
    action_words = ['complaint', 'report', 'file', 'contact', 'regulatory', 
                    'cfpb', 'ftc', 'bbb', 'attorney general']
    features['action_threat'] = sum(1 for word in action_words if word in text_lower)
    
    # Negative sentiment words
    negative_words = ['never', 'nothing', 'no one', 'refuse', 'denied', 
                      'ignore', 'useless', 'incompetent']
    features['negative_score'] = sum(1 for word in negative_words if word in text_lower)
    
    # Money-related (disputes often about money)
    money_words = ['refund', 'money', 'dollar', 'payment', 'charge', 
                   'fee', 'cost', 'price']
    features['money_mention'] = sum(1 for word in money_words if word in text_lower)
    
    return features


def extract_metadata_features(row):
    """
    Extract features from complaint metadata
    
    Args:
        row: DataFrame row with metadata columns
        
    Returns:
        dict: Dictionary of metadata features
    """
    features = {}
    
    # Check if customer is tagged (vulnerable population)
    features['has_tags'] = 1 if pd.notna(row.get('Tags')) else 0
    features['is_older_american'] = 1 if row.get('Tags') == 'Older American' else 0
    features['is_servicemember'] = 1 if row.get('Tags') == 'Servicemember' else 0
    
    # Submission method (some might indicate urgency)
    submission = row.get('Submitted via', '')
    features['submitted_web'] = 1 if submission == 'Web' else 0
    features['submitted_referral'] = 1 if submission == 'Referral' else 0
    features['submitted_phone'] = 1 if submission == 'Phone' else 0
    
    # Product type (some products might have higher escalation rates)
    product = row.get('Product', '')
    features['is_debt_collection'] = 1 if 'Debt collection' in str(product) else 0
    features['is_mortgage'] = 1 if 'Mortgage' in str(product) else 0
    features['is_credit_reporting'] = 1 if 'Credit reporting' in str(product) else 0
    
    return features


def create_feature_matrix(df, text_column='Consumer complaint narrative'):
    """
    Create complete feature matrix combining text and metadata features
    
    Args:
        df: DataFrame with complaints
        text_column: Name of column containing complaint text
        
    Returns:
        DataFrame: Feature matrix
    """
    # Extract text features
    text_features = df[text_column].apply(extract_text_features)
    text_features_df = pd.DataFrame(list(text_features))
    
    # Extract metadata features
    metadata_features = df.apply(extract_metadata_features, axis=1)
    metadata_features_df = pd.DataFrame(list(metadata_features))
    
    # Combine all features
    feature_matrix = pd.concat([text_features_df, metadata_features_df], axis=1)
    
    # Fill any NaN values with 0
    feature_matrix = feature_matrix.fillna(0)
    
    return feature_matrix


if __name__ == "__main__":
    # Example usage
    sample_text = "This company committed FRAUD! I want my money back IMMEDIATELY or I will contact my attorney and file a lawsuit!"
    
    features = extract_text_features(sample_text)
    
    print("Extracted Features:")
    for key, value in features.items():
        if value > 0:  # Only show non-zero features
            print(f"  {key}: {value}")
