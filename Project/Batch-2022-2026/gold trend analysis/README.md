# Gold Price Prediction System

## ⚙️ Requirements

- **Python 3.8 - 3.10** (Tested on Python 3.8)
- Gold price dataset (CSV format)

## 📊 Dataset Format

Place your gold price data in `gold_data.csv` with these columns:
- **Date** - Date in YYYY-MM-DD format
- **Open** - Opening price
- **High** - Highest price
- **Low** - Lowest price
- **Close** - Closing price
- **Volume** - Trading volume

See `gold_data_template.csv` for format example.

**Note:** If no CSV is provided, the system will generate sample data automatically.

## 📊 Required Values for Prediction

The model requires the following data to predict gold prices:

1. **Today's Gold Price** - Current market price (USD per ounce)
2. **Historical Prices** - Last 60 days of OHLC data (Open, High, Low, Close)
3. **Trading Volume** - Market volume data
4. **Technical Indicators** (Auto-calculated):
   - MA 10 (10-day Moving Average)
   - MA 30 (30-day Moving Average)
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)

## 🚀 Setup Instructions

### Step 1: Check Python version
```bash
python check_version.py
```

### Step 2: Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Setup Ollama AI (Optional - for AI features)

**Install Ollama:**
- Download from: https://ollama.ai/download
- Install and restart your computer

**Setup the AI model:**
```bash
setup_ollama.bat
```
Or manually:
```bash
ollama pull llama2
```

**Start Ollama service (keep this running):**
```bash
start_ollama.bat
```
Or manually:
```bash
ollama serve
```

### Step 4: Train the model (first time only)
```bash
python gold_predictor.py
```

### Step 5: Start the Flask server
```bash
python app.py
```

### Step 6: Open your browser
```
http://localhost:5000
```

## 🤖 AI Features (Requires Ollama)

If Ollama is running, you can:
- Ask questions about gold predictions
- Get AI-powered investment analysis
- Receive risk assessments
- Get actionable recommendations

**Note:** The app works without Ollama, but AI features will be disabled.

## 💡 How to Use

1. **Enter Today's Price**: Input the current gold price or leave empty to use latest market data
2. **Click "Predict Tomorrow's Price"**: The ML model will analyze and predict
3. **View Results**: 
   - Direction (UP ⬆️ or DOWN ⬇️)
   - Today's price
   - Predicted price
   - Expected change percentage
   - Confidence level
4. **Analyze Charts**: View historical trends, moving averages, RSI, and MACD indicators

## 🧠 Model Details

- **Algorithm**: LSTM (Long Short-Term Memory) Neural Network
- **Training Data**: 2 years of historical gold prices
- **Features**: 6 technical indicators
- **Sequence Length**: 60 days
- **Prediction**: Next day's closing price

## 📈 Dashboard Features

- Real-time price prediction
- Interactive charts with 6 months of data
- Technical indicator visualization
- Responsive design
- Clear UP/DOWN signals

## ⚠️ Disclaimer

This is a predictive model for educational purposes. Always consult financial advisors before making investment decisions.
