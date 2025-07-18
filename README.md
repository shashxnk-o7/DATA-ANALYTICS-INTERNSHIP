# Sentiment-analysis-nlp
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("products.csv")  # Make sure this file is in the same folder

# Sentiment analysis
df['Sentiment'] = df['Products'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Categorize
df['Sentiment_Label'] = df['Sentiment'].apply(
    lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
)

# Count and plot
sentiment_counts = df['Sentiment_Label'].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['lightgreen', 'lightgrey', 'lightcoral'])
plt.title('Sentiment Distribution of Products')
plt.xlabel('Sentiment')
plt.ylabel('Number of Transactions')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
