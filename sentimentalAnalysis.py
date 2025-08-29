from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

# Dummy data - sample text reviews/comments
dummy_data = [
    "I absolutely love this product! It's amazing and works perfectly.",
    "This is the worst purchase I've ever made. Total waste of money.",
    "It's okay, nothing special but does the job.",
    "Great quality and fast shipping. Highly recommend!",
    "Not what I expected. Pretty disappointed with the results.",
    "Fantastic! Exceeded all my expectations. Will buy again.",
    "Meh, it's average. Neither good nor bad.",
    "Terrible customer service and poor quality product.",
    "Outstanding performance and great value for money!",
    "I'm neutral about this. It's just okay.",
    "Worst experience ever! Would not recommend to anyone.",
    "Excellent product! Really happy with my purchase.",
    "It's decent, could be better but not too bad.",
    "Amazing quality! Worth every penny.",
    "Complete disaster. Nothing worked as promised."
]

def analyze_sentiment(text):
    """
    Analyze sentiment of a given text using TextBlob
    Returns polarity (-1 to 1) and subjectivity (0 to 1)
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def classify_sentiment(polarity):
    """
    Classify sentiment based on polarity score
    """
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Perform sentiment analysis
results = []
for i, text in enumerate(dummy_data, 1):
    polarity, subjectivity = analyze_sentiment(text)
    sentiment = classify_sentiment(polarity)
    
    results.append({
        'id': i,
        'text': text,
        'polarity': polarity,
        'subjectivity': subjectivity,
        'sentiment': sentiment
    })

# Create DataFrame for better visualization
df = pd.DataFrame(results)

# Display results
print("SENTIMENT ANALYSIS RESULTS")
print("=" * 50)
print(f"{'ID':<3} {'Sentiment':<10} {'Polarity':<10} {'Text':<50}")
print("-" * 75)

for _, row in df.iterrows():
    print(f"{row['id']:<3} {row['sentiment']:<10} {row['polarity']:<10.2f} {row['text'][:47]}...")

# Summary statistics
print(f"\nSUMMARY STATISTICS")
print("=" * 30)
sentiment_counts = df['sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{sentiment}: {count} ({percentage:.1f}%)")

print(f"\nAverage Polarity: {df['polarity'].mean():.3f}")
print(f"Average Subjectivity: {df['subjectivity'].mean():.3f}")

# Create visualization
plt.figure(figsize=(12, 5))

# Plot 1: Sentiment Distribution
plt.subplot(1, 2, 1)
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Plot 2: Polarity vs Subjectivity
plt.subplot(1, 2, 2)
colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
for sentiment in df['sentiment'].unique():
    subset = df[df['sentiment'] == sentiment]
    plt.scatter(subset['polarity'], subset['subjectivity'], 
               c=colors[sentiment], label=sentiment, alpha=0.7)

plt.xlabel('Polarity (Negative ← → Positive)')
plt.ylabel('Subjectivity (Objective ← → Subjective)')
plt.title('Polarity vs Subjectivity')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Optional: Save results to CSV
# df.to_csv('sentiment_analysis_results.csv', index=False)
# print("\nResults saved to 'sentiment_analysis_results.csv'")
