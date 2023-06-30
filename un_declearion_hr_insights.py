import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk

# Read the text data from the file
with open("un_declaration_hr_text_data.txt", "r") as file:
    text = file.read()

# Tokenize the text into individual words
words = nltk.word_tokenize(text)

# Define a list of stopwords to be excluded
stopwords = nltk.corpus.stopwords.words("english")

# Remove stopwords from the list of words
words = [word.lower() for word in words if word.isalnum() and word.lower() not in stopwords]

# Generate word frequency counts
word_counts = Counter(words)

# Generate a word cloud of the most frequent terms
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_counts)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud: Most Frequent Terms")
plt.show()

# Get the top 25 frequent terms
top_terms = word_counts.most_common(25)

# Extract the terms and their frequencies
terms, frequencies = zip(*top_terms)

# Plot a bar plot of the top 25 frequent terms
plt.figure(figsize=(12, 6))
plt.bar(terms, frequencies)
plt.xticks(rotation=90)
plt.xlabel("Terms")
plt.ylabel("Frequencies")
plt.title("Top 25 Frequent Terms (Excluding Stopwords)")
plt.tight_layout()
plt.show()