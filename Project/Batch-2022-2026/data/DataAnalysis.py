import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def dataAnalysis():
    data = pd.read_csv('insurance.csv')
    clean_data = {'sex': {'male': 0, 'female': 1},
                  'smoker': {'no': 0, 'yes': 1},
                  'region': {'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3}
                  }
    data_copy = data.copy()
    data_copy.replace(clean_data, inplace=True)

    corr = data_copy.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap='BuPu', annot=True, fmt=".2f", ax=ax)
    plt.title("Dependencies of Medical Charges")
    plt.savefig('static/vis/Cor.jpg')

    plt.figure(figsize=(12, 9))
    plt.title('Age vs Charge')
    sns.barplot(x='age', y='charges', data=data_copy, palette='husl')
    plt.savefig('static/vis/AgevsCharges.jpg')
    plt.clf()

    plt.figure(figsize=(10, 7))
    plt.title('Region vs Charge')
    sns.barplot(x='region', y='charges', data=data_copy, palette='Set3')
    plt.savefig('static/vis/RegvsCharges.jpg')
    plt.clf()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x='bmi', y='charges', hue='sex', data=data_copy, palette='Reds')
    plt.title('BMI VS Charge')
    plt.savefig('static/vis/bmivsCharges.jpg')
    plt.clf()

    plt.figure(figsize=(10, 7))
    plt.title('Smoker vs Charge')
    sns.barplot(x='smoker', y='charges', data=data_copy, palette='Blues', hue='sex')
    plt.savefig('static/vis/smokevsCharges.jpg')
    plt.clf()

    plt.figure(figsize=(10, 7))
    plt.title('Gender vs Charges')
    sns.barplot(x='sex', y='charges', data=data_copy, palette='Set1')
    plt.savefig('static/vis/sexvsCharges.jpg')
    plt.clf()


#dataAnalysis()


