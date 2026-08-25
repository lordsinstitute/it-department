from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from gold_predictor import GoldPredictor
from ollama_assistant import OllamaAssistant
import os

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

predictor = GoldPredictor()
ollama = OllamaAssistant()

@app.route('/')
def index():
    return send_from_directory('templates', 'dashboard.html')

@app.route('/dashboard.html')
def dashboard():
    return send_from_directory('templates', 'dashboard.html')

@app.route('/dataset_info.html')
def dataset_info():
    return send_from_directory('templates', 'dataset_info.html')

@app.route('/model_info.html')
def model_info():
    return send_from_directory('templates', 'model_info.html')

@app.route('/validation.html')
def validation():
    return send_from_directory('templates', 'validation.html')

@app.route('/trends.html')
def trends():
    return send_from_directory('templates', 'trends.html')

@app.route('/ask_ai.html')
def ask_ai():
    return send_from_directory('templates', 'ask_ai.html')

@app.route('/manual_entry.html')
def manual_entry():
    return send_from_directory('templates', 'manual_entry.html')

@app.route('/api/validate', methods=['GET'])
def validate():
    try:
        num_predictions = request.args.get('num', 30, type=int)
        result = predictor.validate_model(num_predictions)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        current_price = data.get('current_price', None)
        
        result = predictor.predict_tomorrow(current_price)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict_manual', methods=['POST'])
def predict_manual():
    try:
        data = request.json
        result = predictor.predict_with_manual_data(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical', methods=['GET'])
def historical():
    try:
        data = predictor.get_historical_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recent_trends', methods=['GET'])
def recent_trends():
    try:
        current_price = request.args.get('price', None, type=float)
        
        # Historical data
        week_data = predictor.get_recent_data(period='1w')
        month_data = predictor.get_recent_data(period='1mo')
        
        # Future predictions
        week_future = predictor.predict_future(num_days=7, current_price=current_price)
        month_future = predictor.predict_future(num_days=30, current_price=current_price)
        
        return jsonify({
            'week': week_data,
            'month': month_data,
            'week_future': week_future,
            'month_future': month_future
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train():
    try:
        predictor.train(epochs=30)
        return jsonify({'status': 'success', 'message': 'Model trained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/exchange_rate', methods=['GET'])
def get_exchange_rate():
    try:
        import requests
        # Fetch live USD to INR rate from exchangerate-api.com (free API)
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=5)
        if response.status_code == 200:
            data = response.json()
            inr_rate = data['rates'].get('INR', 91.0)  # Default to 91 if not found
            return jsonify({'rate': inr_rate, 'source': 'live'})
        else:
            return jsonify({'rate': 91.0, 'source': 'default'})
    except:
        # Fallback to 91 if API fails
        return jsonify({'rate': 91.0, 'source': 'default'})

@app.route('/api/ollama/status', methods=['GET'])
def ollama_status():
    try:
        is_connected = ollama.check_connection()
        return jsonify({'connected': is_connected})
    except Exception as e:
        return jsonify({'connected': False, 'error': str(e)})

@app.route('/api/ollama/analyze', methods=['POST'])
def ollama_analyze():
    try:
        data = request.json
        prediction_data = data.get('prediction_data', {})
        historical_summary = data.get('historical_summary', '')
        
        analysis = ollama.generate_analysis(prediction_data, historical_summary)
        return jsonify({'analysis': analysis})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ollama/ask', methods=['POST'])
def ollama_ask():
    try:
        data = request.json
        question = data.get('question', '')
        context_data = data.get('context_data', {})
        
        answer = ollama.answer_question(question, context_data)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
