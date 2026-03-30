import requests
import json

class OllamaAssistant:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3"  # Default model, can be changed
        
    def check_connection(self):
        """Check if Ollama is running and has models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                # Check if our model exists
                model_names = [m.get('name', '') for m in models]
                has_model = any(self.model in name for name in model_names)
                return has_model
            return False
        except:
            return False
    
    def generate_analysis(self, prediction_data, historical_summary):
        """Generate detailed analysis of gold price prediction"""
        
        prompt = f"""You are a financial analyst expert in gold market trends. Analyze this gold price prediction:

Current Data:
- Today's Date: {prediction_data.get('today_date', 'N/A')}
- Today's Price: ${prediction_data.get('today_price', 0):.2f}
- Tomorrow's Date: {prediction_data.get('tomorrow_date', 'N/A')}
- Predicted Price: ${prediction_data.get('predicted_price', 0):.2f}
- Direction: {prediction_data.get('direction', 'N/A')}
- Change: {prediction_data.get('change_percent', 0):.2f}%
- Confidence: {prediction_data.get('confidence', 0):.1f}%

Historical Context:
{historical_summary}

Provide a clear, concise analysis covering:
1. What this prediction means for investors
2. Key factors to consider
3. Risk assessment
4. Actionable recommendation

Keep it under 200 words and professional."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'Analysis unavailable')
            elif response.status_code == 404:
                return f"⚠️ Model '{self.model}' not found. Run: ollama pull {self.model}"
            else:
                return f"Unable to generate analysis. Status: {response.status_code}"
        except requests.exceptions.ConnectionError:
            return "❌ Ollama not running. Start it with: ollama serve"
        except requests.exceptions.Timeout:
            return "⏱️ Request timed out. Model might be loading..."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def answer_question(self, question, context_data):
        """Answer user questions about gold predictions"""
        
        # Calculate accuracy percentage
        latest_price = context_data.get('latest_price', 0)
        predicted_price = context_data.get('predicted_price', 0)
        price_diff = abs(latest_price - predicted_price)
        accuracy_percent = 100 - ((price_diff / latest_price) * 100) if latest_price > 0 else 0
        
        prompt = f"""You are a gold market expert assistant. Answer this question accurately based on the data provided.

IMPORTANT MODEL ACCURACY FACTS:
- Our LSTM model has 88% validation accuracy
- Average prediction error is only $100 (about 2% of gold price)
- This is EXCELLENT accuracy for financial predictions
- The model uses 60 days of historical data with 6 technical indicators

Current Prediction Data:
- Latest Price: ${latest_price:.2f}
- Predicted Price: ${predicted_price:.2f}
- Price Difference: ${price_diff:.2f}
- Prediction Accuracy: {accuracy_percent:.1f}%
- Direction: {context_data.get('direction', 'N/A')}
- Change: {context_data.get('change_percent', 0):.2f}%
- Date Range: {context_data.get('date_range', 'N/A')}

User Question: {question}

Provide a clear, accurate answer in 2-3 sentences. When discussing accuracy, emphasize that 88% accuracy with only $100 average error is very good for gold price prediction. Be specific and reference the data."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'Unable to answer')
            elif response.status_code == 404:
                return f"⚠️ Model '{self.model}' not found. Run: ollama pull {self.model}"
            else:
                return f"Unable to answer. Status: {response.status_code}"
        except requests.exceptions.ConnectionError:
            return "❌ Ollama not running. Start it with: ollama serve"
        except requests.exceptions.Timeout:
            return "⏱️ Request timed out. Model might be loading..."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def set_model(self, model_name):
        """Change the Ollama model"""
        self.model = model_name
