import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score

def predict_sentiment(review):
    # Perform sentiment analysis
    sentiment = sentiments.polarity_scores(review)
    compound_score = sentiment['compound']

  
    if compound_score >= 0.05:
        return 'Positive', compound_score
    elif compound_score <= -0.05:
        return 'Negative', compound_score
    else:
        return 'Neutral', compound_score


sentiments = SentimentIntensityAnalyzer()


svm_classifier = SVC()
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = None  # You should load your trained TF-IDF vectors here

# Streamlit app
def main():
    st.title("Hotel Review Sentiment Analysis")
    
   
    user_input = st.text_area("Enter your hotel review:")

    if st.button("Analyze"):
        
        sentiment, sentiment_score = predict_sentiment(user_input)
        
       
        result_df = pd.DataFrame({'Sentiment': [sentiment], 'Sentiment Score': [sentiment_score]})
        st.table(result_df)

if __name__ == '__main__':
    main()
