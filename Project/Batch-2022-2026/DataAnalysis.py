import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
#from scipy.stats import chi2_contingency

def dataAnalysis():
    data = pd.read_csv('final_data.csv')
    df = data.drop(
        ['Model Year', 'Make', 'Model', 'Fuel Consumption (City (L/100 km)', 'Fuel Consumption(Hwy (L/100 km))',
         'Fuel Consumption(Comb (mpg))', 'CO2 Emissions(g/km)', 'Smog Rating'], axis=1)

    df = df.rename(columns={'Vehicle Class': 'Vehicle Class', 'Engine Size(L)': 'Engine Size', 'Cylinders': 'Cylinders',
                            'Transmission': 'Transmission', 'Fuel Type': 'Fuel Type',
                            'Fuel Consumption(Comb (L/100 km))': 'Fuel Consumption', 'CO2 Rating': 'CO2 Rating'})
    df['Fuel Type'].fillna((df['Fuel Type'].mode()[0]), inplace=True)

    df['CO2 Rating'].fillna(0, inplace=True)
    new_ratting = []

    for fuel, co2 in zip(df['Fuel Consumption'], df['CO2 Rating']):
        if co2 == 0:
            if 20 <= fuel:
                new_ratting.append(1)
            elif 16.0 <= fuel < 20.0:
                new_ratting.append(2)
            elif 14.0 <= fuel < 16.0:
                new_ratting.append(3)
            elif 12.0 <= fuel < 14.0:
                new_ratting.append(4)
            elif 10.0 <= fuel < 12.0:
                new_ratting.append(5)
            elif 8.0 <= fuel < 10.0:
                new_ratting.append(6)
            elif 7.0 <= fuel < 8.0:
                new_ratting.append(7)
            elif 6.0 <= fuel < 7.0:
                new_ratting.append(8)
            elif 5.0 <= fuel < 6.0:
                new_ratting.append(9)
            elif fuel < 5.0:
                new_ratting.append(10)
        else:
            new_ratting.append(co2)

    df['CO2 Rating'] = new_ratting

    df = df.replace(
        {'Transmission': {'AM8': 'AM', 'AS10': 'AS', 'A8': 'A', 'A9': 'A', 'AM7': 'AM', 'AS8': 'AS', 'M6': 'M', \
                          'AS6': 'AS', 'AS9': 'AS', 'A10': 'A', 'A6': 'A', 'M5': 'M', 'M7': 'M', 'AV7': 'AV',
                          'AV1': 'AV', 'AM6': 'AM', 'AS7': 'AS', 'AV8': 'AV', 'AV6': 'AV', 'AV10': 'AV', 'AS5': 'AS',
                          'A7': 'A'}})

    plt.figure(figsize=(13, 6), dpi=150)
    ax = sns.histplot(data=df, x='Transmission', color='DarkOliveGreen')

    # Loop through bars and add labels
    for p in ax.patches:
        ax.annotate(
            str(int(p.get_height())),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=10
        )

    plt.savefig('static/vis/trans_dist.jpg')
    plt.clf()

    plt.figure(figsize=(5, 4), dpi=150)
    chart1 = sns.histplot(data=df, x='Fuel Type', color='PaleVioletRed')
    plt.savefig('static/vis/ftype_dist.jpg')
    plt.clf()

    plt.figure(figsize=(13, 6), dpi=150)
    plt.xticks(rotation=45)
    plt.title('Cylinders vs Consumption', size=20)
    chart1 = sns.barplot(data=df, x="Cylinders", y="Fuel Consumption", palette='mako_r', ci=None)
    plt.xlabel('Cylinders', size=20)
    plt.ylabel('Fuel Consumption', size=20)
    chart1.bar_label(chart1.containers[0], size=12)
    plt.savefig('static/vis/cyl_cns.jpg')
    plt.clf()


    df_numeric = df.select_dtypes(include=['number'])
    correlation_matrix = df_numeric.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='YlGn', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Fuel Consumption Dataset")
    plt.savefig('static/vis/corr.jpg')
    plt.clf()

    vehicle_class_consumption = df.groupby("Vehicle Class")["Fuel Consumption"].mean().sort_values()
    plt.figure(figsize=(12, 6))
    vehicle_class_consumption.plot(kind="bar", color="skyblue")
    plt.xlabel("Vehicle Class")
    plt.ylabel("Average Fuel Consumption (L/100 km)")
    plt.title("Average Fuel Consumption by Vehicle Class")
    plt.xticks(rotation=100)
    plt.savefig('static/vis/veh_cls.jpg')
    plt.clf()


#dataAnalysis()



