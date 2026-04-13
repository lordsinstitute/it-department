# EDA imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


dictionary = {
        "FAVC": "Frequent consumption of high caloric food",
        "FCVC": "Frequency of consumption of vegetables",
        "NCP": "Number of main meals",
        "CAEC": "Consumption of food between meals",
        "CH20": "Consumption of water daily",
        "CALC": "Consumption of alcohol",
        "SCC": "Calories consumption monitoring",
        "FAF": "Physical activity frequency",
        "TUE": "Time using technology devices",
        "MTRANS": "Transportation used",
    }

def dataAnalysis():
    train_df1 = pd.read_csv('train.csv')
    train_df2 = pd.read_csv('ObesityDataSet.csv')
    test_df = pd.read_csv('test.csv')

    full_train = pd.concat([train_df1, train_df2], axis=0).reset_index(drop=True)
    full_train.drop('id', axis=1, inplace=True)
    full_train.insert(0, 'id', full_train.index + 1)
    full_train.rename(columns={'family_history_with_overweight': 'FamHist'}, inplace=True)
    test_df.rename(columns={'family_history_with_overweight': 'FamHist'}, inplace=True)

    plt.figure(figsize=(14, 6))
    ax = sns.countplot(x="NObeyesdad", data=full_train, palette="husl")
    # ax.bar_label(ax.containers[0])
    plt.xticks(rotation=45)
    ax.set_title(f"Obesity target distribution in {full_train.shape[0]} samples", fontsize=14)
    # Annotate bars with their counts
    for p in ax.patches:
        plt.annotate(f"\n\n{int(p.get_height())}", (p.get_x() + 0.4, p.get_height()), ha='center', va='top',
                     color='white', size=10)
    plt.savefig('static/vis/OTD.jpg')
    plt.clf()

    plt.figure(figsize=(14, 6))
    ax = sns.countplot(x="Gender", data=full_train, hue="NObeyesdad", palette="husl")
    # ax.bar_label(ax.containers[0])
    plt.xticks(rotation=45)
    ax.set_title(f"Obesity target distribution in Gender", fontsize=14)
    # Annotate bars with their counts
    for bar in plt.gca().patches:
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 str(bar.get_height()),
                 ha='center', va='bottom')

    plt.savefig('static/vis/gender.jpg')
    plt.clf()

    plt.figure(figsize=(14, 6))
    ax = sns.countplot(x="FamHist", data=full_train, hue="NObeyesdad", palette="husl")
    # ax.bar_label(ax.containers[0])
    plt.xticks(rotation=45)
    ax.set_title(f"Obesity target distribution in Family History", fontsize=14)
    # Annotate bars with their counts
    for bar in plt.gca().patches:
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 str(bar.get_height()),
                 ha='center', va='bottom')

    plt.savefig('static/vis/FamHist.jpg')
    plt.clf()

    plt.figure(figsize=(14, 6))
    ax = sns.countplot(x="FAVC", data=full_train, hue="NObeyesdad", palette="husl")
    # ax.bar_label(ax.containers[0])
    plt.xticks(rotation=45)
    ax.set_title(f"Obesity target distribution in Frequent consumption of high caloric food", fontsize=14)
    # Annotate bars with their counts
    for bar in plt.gca().patches:
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 str(bar.get_height()),
                 ha='center', va='bottom')

    plt.savefig('static/vis/FAVC.jpg')
    plt.clf()

    plt.figure(figsize=(14, 6))
    ax = sns.countplot(x="SCC", data=full_train, hue="NObeyesdad", palette="husl")
    # ax.bar_label(ax.containers[0])
    plt.xticks(rotation=45)
    ax.set_title(f"Obesity target distribution in Calories consumption monitoring", fontsize=14)
    # Annotate bars with their counts
    for bar in plt.gca().patches:
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 str(bar.get_height()),
                 ha='center', va='bottom')

    plt.savefig('static/vis/SCC.jpg')
    plt.clf()




#dataAnalysis()


