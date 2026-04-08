from flask import Flask, render_template, request, jsonify
from dataclasses import dataclass
from typing import Dict, Optional
from pydantic import BaseModel, Field
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
from google import genai
import os
import time
import json

# -------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------
load_dotenv()
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# -------------------------------------------------------------
# Flask app setup
# -------------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------------
# Models and Data Classes
# -------------------------------------------------------------
class AQIResponse(BaseModel):
    success: bool
    data: Dict[str, float]
    status: str
    expiresAt: str

class ExtractSchema(BaseModel):
    aqi: float = Field(description="Air Quality Index")
    temperature: float = Field(description="Temperature in degrees Celsius")
    humidity: float = Field(description="Humidity percentage")
    wind_speed: float = Field(description="Wind speed in kilometers per hour")
    pm25: float = Field(description="Particulate Matter 2.5 micrometers")
    pm10: float = Field(description="Particulate Matter 10 micrometers")
    co: float = Field(description="Carbon Monoxide level")

@dataclass
class UserInput:
    city: str
    state: str
    country: str
    medical_conditions: Optional[str]
    planned_activity: str


# -------------------------------------------------------------
# AQI Analyzer Class
# -------------------------------------------------------------
class AQIAnalyzer:
    def __init__(self, firecrawl_key: str) -> None:
        self.firecrawl = FirecrawlApp(api_key=firecrawl_key)

    def _format_url(self, country: str, state: str, city: str) -> str:
        country_clean = country.lower().replace(' ', '-')
        city_clean = city.lower().replace(' ', '-')
        if not state or state.lower() == 'none':
            return f"https://www.aqi.in/dashboard/{country_clean}/{city_clean}"
        state_clean = state.lower().replace(' ', '-')
        return f"https://www.aqi.in/dashboard/{country_clean}/{state_clean}/{city_clean}"

    def fetch_aqi_data(self, city: str, state: str, country: str) -> Dict[str, float]:
        try:
            url = self._format_url(country, state, city)
            start_time = time.perf_counter()

            response = self.firecrawl.extract(
                urls=[url],
                prompt="Extract the current AQI, temperature, humidity, wind speed, PM2.5, PM10, and CO levels from the page.",
                schema=ExtractSchema.model_json_schema()
            )

            elapsed = time.perf_counter() - start_time
            print(f"[INFO] Firecrawl took {elapsed:.2f}s for {url}")

            # 🔥 Correct handling for ExtractResponse object
            if not response.success:
                raise ValueError(f"Failed to fetch AQI data: {response.status}")

            return response.data

        except Exception as e:
            print(f"[ERROR] Firecrawl failed: {e}")
            return {
                'aqi': 0,
                'temperature': 0,
                'humidity': 0,
                'wind_speed': 0,
                'pm25': 0,
                'pm10': 0,
                'co': 0
            }


# -------------------------------------------------------------
# Health Recommendation Agent (Gemini 2.5 Flash - NEW SDK)
# -------------------------------------------------------------
# -------------------------------------------------------------
# Health Recommendation Agent (Gemini 2.5 Flash - JSON Output)
# -------------------------------------------------------------
class HealthRecommendationAgent:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def get_recommendations(self, aqi_data, user_input):

        prompt = f"""
You are an environmental health expert.

Return your response STRICTLY in this JSON format:

{{
  "impact": "Detailed health impact explanation",
  "precautions": "Clear safety precautions",
  "advisable": "Is the activity advisable or not",
  "timing": "Best suggested timing for activity"
}}

Air Quality Data:
AQI: {aqi_data['aqi']}
PM2.5: {aqi_data['pm25']}
PM10: {aqi_data['pm10']}
CO: {aqi_data['co']}
Temperature: {aqi_data['temperature']}
Humidity: {aqi_data['humidity']}
Wind Speed: {aqi_data['wind_speed']}

User Information:
Medical Conditions: {user_input.medical_conditions or 'None'}
Planned Activity: {user_input.planned_activity}

Important:
- Return ONLY valid JSON
- Do not include markdown
- Do not include explanations outside JSON
"""

        start_time = time.perf_counter()

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": 0.6,
                "max_output_tokens": 1500
            }
        )

        print(f"[INFO] Gemini 2.5 Flash took {time.perf_counter() - start_time:.2f}s")

        # -------------------------------
        # Safe JSON extraction
        # -------------------------------
        raw_text = response.text.strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback if model returns unexpected formatting
            return {
                "impact": raw_text,
                "precautions": "",
                "advisable": "",
                "timing": ""
            }


# -------------------------------------------------------------
# Route Handlers
# -------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


import json


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        city = request.form['city']
        state = request.form.get('state', '')
        country = request.form.get('country', 'India')
        medical_conditions = request.form.get('medical_conditions', '')
        planned_activity = request.form['planned_activity']

        user_input = UserInput(city, state, country, medical_conditions, planned_activity)
        aqi_analyzer = AQIAnalyzer(FIRECRAWL_API_KEY)
        health_agent = HealthRecommendationAgent()

        print("[INFO] Fetching AQI data...")
        aqi_data = aqi_analyzer.fetch_aqi_data(city, state, country)

        print("[INFO] Getting recommendations...")

        # 🔥 DO NOT json.loads again
        recommendation_sections = health_agent.get_recommendations(aqi_data, user_input)

        return render_template(
            'results.html',
            aqi_data=aqi_data,
            recommendation_sections=recommendation_sections,
            aqi=aqi_data.get("aqi", 0),
            status="Air Quality Analysis",
            city=city,
            state=state,
            country=country
        )

    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({'error': str(e)}), 500


# -------------------------------------------------------------
# Run Server
# -------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)