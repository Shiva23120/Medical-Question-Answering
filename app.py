import streamlit as st
from transformers import pipeline
import pickle

# Load the pre-trained model
with open('NLP.pkl', 'rb') as f:
    model = pickle.load(f)

# Load a text generation model pipeline
gen_pipeline = pipeline("text-generation", model="gpt2")

# Function to generate an answer using the model
def generate_answer(question):
    result = gen_pipeline(question, max_length=100, num_return_sequences=1)
    return result[0]['generated_text']

# Streamlit app structure
st.title("Medical Question Answering System")

# Instructions
st.write("Enter your medical question below, and the model will provide an answer.")

# Input: User question
user_question = st.text_input("Your Question:")

# Button to submit the question
if st.button("Get Answer"):
    # Check if a question was provided
    if user_question:
        # Generate answer using the model
        answer = generate_answer(user_question)
        
        # Display the answer
        st.write("**Answer:**", answer)
    else:
        st.write("Please enter a question to get an answer.")
