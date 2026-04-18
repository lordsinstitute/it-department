import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # IMPORTANT for Flask / servers
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
from sklearn.metrics import accuracy_score
from keras import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import math
from sklearn.metrics import r2_score
import json

# -----------------------------
# Create folder
# -----------------------------
output_dir = "../static/lstm"
os.makedirs(output_dir, exist_ok=True)
model_dir = "../models/lstm"
os.makedirs(model_dir, exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv('../bitcoin_last_10_years.csv')

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
ds = min_max_scaler.fit_transform(df['Price'].values.reshape(-1, 1))

train_size = int(len(ds) * 0.7)
train = ds[:train_size]
test = ds[train_size:]

look_back = 15

# -----------------------------
# Prepare Train Data
# -----------------------------
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(look_back, len(dataset)):
        dataX.append(dataset[i-look_back:i])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)

x_train, y_train = create_dataset(train, look_back)
x_test, y_test = create_dataset(test, look_back)

# -----------------------------
# Build Model
# -----------------------------
model = Sequential()
model.add(LSTM(150, activation='tanh', input_shape=(x_train.shape[1], 1)))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50,
    batch_size=16,
    verbose=1
)

# -----------------------------
# Save Model
# -----------------------------
model.save(f"{model_dir}/lstm_model.h5")

# -----------------------------
# Accuracy & Loss Curve
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train Loss", "Validation Loss"])
plt.savefig(f"{output_dir}/loss_curve.png")
plt.close()

# -----------------------------
# Predictions
# -----------------------------
trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)

trainPredict = min_max_scaler.inverse_transform(trainPredict)
trainY = min_max_scaler.inverse_transform([y_train]).reshape(y_train.shape)

testPredict = min_max_scaler.inverse_transform(testPredict)
testY = min_max_scaler.inverse_transform([y_test]).reshape(y_test.shape)

# -----------------------------
# RMSE
# -----------------------------
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
testScore = math.sqrt(mean_squared_error(testY, testPredict))

print('Train RMSE: %.2f' % trainScore)
print('Test RMSE: %.2f' % testScore)

# -----------------------------
# R2 Score
# -----------------------------
r2 = r2_score(testY, testPredict)

plt.figure()
plt.bar(["R2 Score"], [r2])
plt.title("R2 Score")
plt.savefig(f"{output_dir}/r2_score.png")
plt.close()

# -----------------------------
# Variance Score
# -----------------------------
variance = np.var(testY - testPredict)

plt.figure()
plt.bar(["Variance"], [variance])
plt.title("Prediction Variance")
plt.savefig(f"{output_dir}/variance.png")
plt.close()

# -----------------------------
# ROC Curve (Directional)
# -----------------------------
# Convert regression to classification (Up/Down)
y_true_direction = (testY[1:] > testY[:-1]).astype(int)
y_pred_direction = (testPredict[1:] > testPredict[:-1]).astype(int)

fpr, tpr, _ = roc_curve(y_true_direction, y_pred_direction)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1])
plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig(f"{output_dir}/roc_curve.png")
plt.close()

# -----------------------------
# Forecast Plot
# -----------------------------
trainPredictPlot = np.empty_like(ds)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

testPredictPlot = np.empty_like(ds)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(ds), :] = testPredict

plt.figure(figsize=(15,7))
plt.plot(min_max_scaler.inverse_transform(ds))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.legend(["Actual Price", "Train Prediction", "Test Prediction"])
plt.ylabel('Price')
plt.xlabel('Day')
plt.savefig(f"{output_dir}/forecast.png")
plt.close()


# R2 Scores
r2_train = r2_score(trainY, trainPredict)
r2_test = r2_score(testY, testPredict)

# Variance Scores
var_train = explained_variance_score(trainY, trainPredict)
var_test = explained_variance_score(testY, testPredict)

metrics = {
    "r2_score": {
        "train": float(r2_train),
        "test": float(r2_test)
    },
    "explained_variance": {
        "train": float(var_train),
        "test": float(var_test)
    }
}

with open("../static/lstm/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Metrics saved as metrics.json")

with open("../static/lstm/metrics.txt", "w") as f:
    f.write(f"R2 Train: {r2_train}\n")
    f.write(f"R2 Test: {r2_test}\n")
    f.write(f"Variance Train: {var_train}\n")
    f.write(f"Variance Test: {var_test}\n")

print("✅ Metrics saved as metrics.txt")

print("✅ All plots saved in static/lstm/")
print("✅ Model saved as lstm_model.h5")