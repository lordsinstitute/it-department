import time, os
from datetime import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dataAnalysis():
    data = pd.read_csv('RTA_Dataset.csv')
    count_donut(data, 'Accident_severity')

    data['Time'] = data['Time'].apply(get_hour)
    data['Time'] = data['Time'].apply(convert_time)

    # Variables except 'Accident_severity' and 'Time'
    features_others = [x for x in data.columns.tolist() if x not in ['Accident_severity', 'Time']]

    # List of features with horizontal xtickmarks (for others we shall make it vertical for visualization convenience)
    features_horiz = ['Sex_of_driver', 'Vehicle_driver_relation', 'Defect_of_vehicle', 'Number_of_vehicles_involved',
                      'Number_of_casualties', 'Sex_of_casualty', 'Casualty_severity']

    # Catplot to compare frequency distributions of features (except 'Time') across target classes

    plt.figure(figsize=(10, 5))
    catplot = sns.catplot(data=data, x='Day_of_week', col='Accident_severity', kind='count', sharey=False)
    #catplot.set_xticklabels(rotation=90)
    plt.suptitle("Frequency distribution of {} by target class".format('Day_of_week'), y=1.1, fontsize=15)
    plt.savefig('static/pimg/fdd.jpg')

    plt.figure(figsize=(10, 5))
    catplot = sns.catplot(data=data, x='Age_band_of_driver', col='Accident_severity', kind='count', sharey=False)
    #catplot.set_xticklabels(rotation=90)
    plt.suptitle("Frequency distribution of {} by target class".format('Age_band_of_driver'), y=1.1, fontsize=15)
    plt.savefig('static/pimg/fda.jpg')

    plt.figure(figsize=(10, 5))
    catplot = sns.catplot(data=data, x='Sex_of_driver', col='Accident_severity', kind='count', sharey=False)
    #catplot.set_xticklabels(rotation=90)
    plt.suptitle("Frequency distribution of {} by target class".format('Sex_of_driver'), y=1.1, fontsize=15)
    plt.savefig('static/pimg/fdg.jpg')

    plt.figure(figsize=(10, 5))
    catplot = sns.catplot(data=data, x='Driving_experience', col='Accident_severity', kind='count', sharey=False)
    #catplot.set_xticklabels(rotation=90)
    plt.suptitle("Frequency distribution of {} by target class".format('Driving_experience'), y=1.1, fontsize=15)
    plt.savefig('static/pimg/fdde.jpg')

    plt.figure(figsize=(10, 5))
    catplot = sns.catplot(data=data, x='Service_year_of_vehicle', col='Accident_severity', kind='count', sharey=False)
    #catplot.set_xticklabels(rotation=90)
    plt.suptitle("Frequency distribution of {} by target class".format('Service_year_of_vehicle'), y=1.1, fontsize=15)
    plt.savefig('static/pimg/fdsy.jpg')


# Add annotations
def add_annotations(ax):
    for p in ax.patches:
        frequency = p.get_height()
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        ax.annotate(f'{frequency}', (x, y), size = 12, ha = 'center', va = 'bottom')


# Visualization
def count_donut(data, col):
    plt.figure(figsize = (14, 7))

    # Countplot
    ax1 = plt.subplot(1, 2, 1)
    count = sns.countplot(x = data[col])
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    sns.despine(top = True, right = True)
    add_annotations(ax1)

    # Donutplot
    ax2 = plt.subplot(1, 2, 2)
    plt.pie(data[col].value_counts(),
            labels = data[col].unique().tolist(),
            autopct = '%1.2f%%',
            pctdistance = 0.8,
            shadow = False,
            radius = 1.3,
            textprops = {'fontsize' : 14}
            )
    circle = plt.Circle((0, 0), 0.4, fc = 'white')
    fig = plt.gcf()
    fig.gca().add_artist(circle)
    ax2.set_xlabel('')

    plt.suptitle("Frequency Comparison of {}".format(col), fontsize = 16)
    plt.subplots_adjust(wspace = 0.4)
    plt.savefig('static/pimg/fcas.jpg')


def get_hour(time):
    value = datetime.strptime(time, '%H:%M:%S')
    return value.hour



def convert_time(time):
    if time >=6 and time <=18:
        return 'Day'
    else:
        return 'Night'

#dataAnalysis()