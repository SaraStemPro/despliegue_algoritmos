from flask import Flask, jsonify, request, render_template, send_from_directory
from transformers import pipeline
import pandas as pd
from pyngrok import ngrok, conf
import threading
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Token de ngrok
load_dotenv('/crendenciales.env')
ngrok_auth_token = os.getenv('ngrok_token')
conf.get_default().auth_token = ngrok_auth_token

# Resultados csv de los modelos del ejercicio anterior
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

@app.route('/api/v1/hello', methods=['GET'])
def hello_world():
    return jsonify(message="Hello, World!")

@app.route('/api/v1/goodbye', methods=['GET'])
def goodbye_world():
    return jsonify(message="Goodbye, World!")

@app.route('/api/v1/time', methods=['GET'])
def get_time():
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return jsonify(time=current_time)

# Pipelines de Hugging Face

@app.route('/api/v1/sentiment', methods=['POST'])
def sentiment_analysis():
    content = request.json
    text = content['text']
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)
    return jsonify(result=result)

@app.route('/api/v1/generate', methods=['POST'])
def text_generation():
    content = request.json
    prompt = content['prompt']
    generator_pipeline = pipeline("text-generation")
    result = generator_pipeline(prompt, max_length=50)
    return jsonify(result=result)

# Métodos adicionales para mostrar resultados en la web

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sentiment')
def sentiment_form():
    return render_template('sentiment.html')

@app.route('/sentiment_result', methods=['POST'])
def sentiment_result():
    text = request.form['text']
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)
    return render_template('sentiment_result.html', result=result[0])

@app.route('/generate')
def generate_form():
    return render_template('generate.html')

@app.route('/generate_result', methods=['POST'])
def generate_result():
    prompt = request.form['prompt']
    generator_pipeline = pipeline("text-generation")
    result = generator_pipeline(prompt, max_length=50)
    return render_template('generate_result.html', result=result[0]['generated_text'])

# Mostrar reportes de los modelos del ejercicio anterior

@app.route('/logistic_regression_report')
def logistic_regression_report():
    report = pd.read_csv(os.path.join(
        DATA_DIR, 'LogisticRegression_classification_report.csv'))
    return render_template('report.html', model_name='Logistic Regression', report=report.to_html())

@app.route('/kneighbors_report')
def kneighbors_report():
    report = pd.read_csv(os.path.join(
        DATA_DIR, 'KNeighborsClassifier_classification_report.csv'))
    return render_template('report.html', model_name='KNeighbors', report=report.to_html())

@app.route('/data/<path:filename>')
def download_file(filename):
    return send_from_directory(DATA_DIR, filename)

# Conectarnos a ngrok
def run_ngrok():
    public_url = ngrok.connect(8000)
    print(" * ngrok tunnel URL:", public_url)

if __name__ == '__main__':
    # He buscado en internet y he visto que con esta librería solucionaba un problema que tenía para conectarme a ngrok
    threading.Thread(target=run_ngrok).start()
    app.run(debug=True, host='0.0.0.0', port=8000)
