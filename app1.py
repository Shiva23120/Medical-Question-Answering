import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.model_selection import train_test_split

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
file_path = 'train.csv'  # Replace with your dataset path
df = pd.read_csv(file_path)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    words = word_tokenize(text)
    cleaned_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words and word not in string.punctuation]
    return ' '.join(cleaned_words)

# Clean the 'Answer' column
df['Cleaned_Answer'] = df['Answer'].apply(clean_text)

# Save the cleaned dataset
cleaned_file_path = 'train_NLP1.csv'
df.to_csv(cleaned_file_path, index=False)

# Load your cleaned dataset
df1 = pd.read_csv(cleaned_file_path)

# Preprocess and vectorize the answers
vectorizer = TfidfVectorizer(stop_words='english')
answers_tfidf = vectorizer.fit_transform(df1['Cleaned_Answer'])  # Vectorize cleaned answers

# Split the data into training and testing sets
X = df1["Question"]
y = df1["Cleaned_Answer"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to retrieve the closest answer
def get_closest_answer(question, similarity_threshold=0.2):
    """
    Returns the closest matching answer based on cosine similarity.
    Includes a similarity threshold to avoid irrelevant answers.
    """
    # Clean the input question
    question_cleaned = clean_text(question)
    
    # Vectorize the question
    question_tfidf = vectorizer.transform([question_cleaned])
    
    # Compute cosine similarity between the question and all answers
    similarities = cosine_similarity(question_tfidf, answers_tfidf)
    
    # Get the closest match
    closest_index = np.argmax(similarities)
    closest_similarity = similarities[0][closest_index]
    
    # Check if the closest match passes the similarity threshold
    if closest_similarity >= similarity_threshold:
        return df1['Cleaned_Answer'].iloc[closest_index]
    else:
        return "Sorry, I couldn't find a relevant answer."

# Example usage
user_question = input("Enter your question: ")
closest_answer = get_closest_answer(user_question)
print(f"Closest Answer: {closest_answer}")


import pickle

# Assuming the processed data `data` (the DataFrame with tokenized and cleaned text) is ready to be saved

# Save the processed dataset to a pickle file
with open(r'C:\Users\msk23\OneDrive\Desktop\NIT\NLP\New folder\processed_data1.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Processed data has been saved to a pickle file.")
