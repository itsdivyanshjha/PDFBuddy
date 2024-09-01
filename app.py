import json
from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
import requests

app = Flask(__name__)

# Ollama API endpoint
OLLAMA_API_URL = "http://44.220.147.121:11434/api/generate"  # Pointing to the Ollama server

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'pdf-file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    pdf_file = request.files['pdf-file']
    pdf_reader = PdfReader(pdf_file)
    pdf_text = ""

    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

    return jsonify({"message": "PDF processed successfully", "pdf_text": pdf_text}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    pdf_text = data.get('pdf_text', '')
    user_question = data.get('question', '')

    if not pdf_text or not user_question:
        return jsonify({"error": "Missing PDF context or question"}), 400

    # Prepare the prompt for Ollama
    prompt = f"{pdf_text}\n\nUser Question: {user_question}\nAnswer:"

    # Prepare the request payload with the required model parameter
    payload = {
        "model": "llama3.1",  # Ensure this matches the model you are using
        "prompt": prompt
    }

    try:
        # Send the prompt to Ollama and handle streaming response
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
        response.raise_for_status()

        # Print the raw response for debugging
        print("Raw response from Ollama API:")
        print(response.text)

        # Accumulate the full response from the stream
        full_response = ""
        for line in response.iter_lines():
            if line:
                line_data = line.decode('utf-8')
                try:
                    json_data = json.loads(line_data)
                    full_response += json_data.get('response', '')
                except json.JSONDecodeError as json_err:
                    print(f"JSON decode error: {json_err}")
                    print(f"Problematic line data: {line_data}")

    except requests.exceptions.RequestException as e:
        print("Error during Ollama API call:", e)
        return jsonify({"error": "Error during Ollama API call"}), 500

    # Return the response as HTML to preserve bold formatting
    return jsonify({"answer": full_response.strip()}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
