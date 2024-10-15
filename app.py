from flask import Flask, render_template, request
from transformers import pipeline
import pickle

# Initialize Flask app
app = Flask(__name__)


with open('NLP.pkl', 'rb') as f:
    model = pickle.load(f)

# Load a text generation model pipeline
# Here, we're using a GPT-based model for generating answers from questions
gen_pipeline = pipeline("text-generation", model="gpt2")

# Function to generate an answer using the model
def generate_answer(question):
    result = gen_pipeline(question, max_length=100, num_return_sequences=1)
    return result[0]['generated_text']

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for processing user input
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        user_question = request.form['question']
        
        # Pass the question to the model to generate an answer
        answer = generate_answer(user_question)
        
        # Return the result to the webpage
        return render_template('index.html', question=user_question, answer=answer)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
