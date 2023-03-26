import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the Twitter dataset
df = pd.read_csv("twitter_sentiment_analysis2.csv")

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to perform sentiment analysis on a given text
def analyze_sentiment(text):
    score = sid.polarity_scores(text)
    return score["compound"]

# Apply the sentiment analyzer to the tweets in the dataset
df["sentiment_score"] = df["text"].apply(analyze_sentiment)

# Classify the tweets as positive, negative, or neutral based on the sentiment score
df["sentiment"] = df["sentiment_score"].apply(lambda score: "positive" if score > 0 else ("negative" if score < 0 else "neutral"))

# Print the percentage of positive, negative, and neutral tweets in the dataset
print(df["sentiment"].value_counts(normalize=True))
