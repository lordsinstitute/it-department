import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# -----------------------------
# Create static/vis directory
# -----------------------------
output_dir = "../static/vis"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------
rawData1 = pd.read_csv('../dataset/visualization.csv', nrows=3)
cols = rawData1.columns
rawData2 = pd.read_csv('../dataset/visualization.csv', skiprows=40254)
rawData2.columns = cols
data = pd.concat([rawData1, rawData2], ignore_index=True)

# =============================
# 1. Consumers With Fraud
# =============================
fig, axs = plt.subplots(2, 1)
fig.suptitle('Consumers With Fraud', fontsize=18)
plt.subplots_adjust(hspace=0.8)

data.loc[0].plot(ax=axs[0], color='firebrick', grid=True)
axs[0].set_title('Consumer 0', fontsize=16)
axs[0].set_xlabel('Dates of Consumption')
axs[0].set_ylabel('Consumption')

data.loc[2].plot(ax=axs[1], color='firebrick', grid=True)
axs[1].set_title('Consumer 1', fontsize=16)
axs[1].set_xlabel('Dates of Consumption')
axs[1].set_ylabel('Consumption')

fig.savefig(os.path.join(output_dir, "consumers_with_fraud.png"))
plt.close(fig)

# =============================
# 2. Consumers Without Fraud
# =============================
fig, axs = plt.subplots(2, 1)
fig.suptitle('Consumers Without Fraud', fontsize=18)
plt.subplots_adjust(hspace=0.8)

data.loc[3].plot(ax=axs[0], color='teal', grid=True)
axs[0].set_title('Consumer 40255', fontsize=16)

data.loc[4].plot(ax=axs[1], color='teal', grid=True)
axs[1].set_title('Consumer 40256', fontsize=16)

fig.savefig(os.path.join(output_dir, "consumers_without_fraud.png"))
plt.close(fig)

# =============================
# 3. Statistics - Without Fraud
# =============================
fig2, axs2 = plt.subplots(2, 2)
fig2.suptitle('Statistics for Consumer 0', fontsize=18)
plt.subplots_adjust(hspace=0.8)

data.loc[0].plot(ax=axs2[0, 0], color='firebrick', grid=True)
axs2[0, 0].set_title('Consumption')

data.loc[0].hist(color='firebrick', ax=axs2[0, 1], grid=True)
axs2[0, 1].set_title('Histogram')

data.loc[0].plot.kde(color='firebrick', ax=axs2[1, 0], grid=True)
axs2[1, 0].set_title('Density')

data.loc[0].describe().drop(['count']).plot(
    kind='bar', ax=axs2[1, 1], color='firebrick', grid=True
)
axs2[1, 1].set_title('Statistics')

fig2.savefig(os.path.join(output_dir, "stats_consumer_0.png"))
plt.close(fig2)

# =============================
# 4. Statistics - With Fraud
# =============================
fig3, axs3 = plt.subplots(2, 2)
fig3.suptitle('Statistics for Consumer 40256', fontsize=18)
plt.subplots_adjust(hspace=0.8)

data.loc[4].plot(ax=axs3[0, 0], color='teal', grid=True)
axs3[0, 0].set_title('Consumption')

data.loc[4].hist(color='teal', ax=axs3[0, 1])
axs3[0, 1].set_title('Histogram')

data.loc[4].plot.kde(color='teal', ax=axs3[1, 0], grid=True)
axs3[1, 0].set_title('Density')

data.loc[4].describe().drop(['count']).plot(
    kind='bar', ax=axs3[1, 1], color='teal', grid=True
)
axs3[1, 1].set_title('Statistics')

fig3.savefig(os.path.join(output_dir, "stats_consumer_40256.png"))
plt.close(fig3)

# =============================
# 5. Four Week Consumption
# =============================
fig4, axs4 = plt.subplots(2, 1)
fig4.suptitle('Four Week Consumption', fontsize=16)
plt.subplots_adjust(hspace=0.5)

for i in range(59, 83, 7):
    axs4[0].plot(data.iloc[0, i:i + 7].to_numpy(),
                 marker='>', linestyle='-',
                 label=f'week {(i - 59)//7 + 1}')
axs4[0].legend()
axs4[0].set_title('Without Fraud')
axs4[0].grid(True)

for i in range(59, 83, 7):
    axs4[1].plot(data.iloc[4, i:i + 7].to_numpy(),
                 marker='>', linestyle='-',
                 label=f'week {(i - 59)//7 + 1}')
axs4[1].legend()
axs4[1].set_title('With Fraud')
axs4[1].grid(True)

fig4.savefig(os.path.join(output_dir, "four_week_consumption.png"))
plt.close(fig4)

# =============================
# 6. Correlation Heatmap
# =============================
fig5, axs5 = plt.subplots(1, 2)

alpha = ['week 1', 'week 2', 'week 3', 'week 4']

# Without Fraud
a = []
for i in range(59, 83, 7):
    a.append(data.iloc[0, i:i + 7].to_numpy())
cor = pd.DataFrame(a).transpose().corr()
cax = axs5[0].matshow(cor)
for (i, j), z in np.ndenumerate(cor):
    axs5[0].text(j, i, f'{z:0.1f}', ha='center', va='center', color='white')
axs5[0].set_xticklabels([''] + alpha)
axs5[0].set_yticklabels([''] + alpha)
axs5[0].set_title('Customer without Fraud')

# With Fraud
a = []
for i in range(59, 83, 7):
    a.append(data.iloc[4, i:i + 7].to_numpy())
cor = pd.DataFrame(a).transpose().corr()
cax = axs5[1].matshow(cor)
for (i, j), z in np.ndenumerate(cor):
    axs5[1].text(j, i, f'{z:0.1f}', ha='center', va='center', color='white')
axs5[1].set_xticklabels([''] + alpha)
axs5[1].set_yticklabels([''] + alpha)
axs5[1].set_title('Customer with Fraud')

fig5.colorbar(cax)
fig5.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close(fig5)

print("All plots saved successfully in static/vis/")
