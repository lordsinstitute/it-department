import os
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def dataAnalysis():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.join(BASE_DIR, "..")
    VIS_DIR = os.path.join(PROJECT_DIR, "static", "vis")

    # Ensure the output directory exists
    os.makedirs(VIS_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(PROJECT_DIR, "fetal_health.csv"))

    # Fix column names: strip whitespace and replace spaces with underscores
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    df["fetal_health"].value_counts(1).plot(kind="bar", color=["yellowgreen", "gold", "firebrick"])

    height = (df["fetal_health"].value_counts(1) * 100).to_list()
    bars = ('Normal', 'Suspect', 'Pathologic')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color=('yellowgreen', 'gold', 'firebrick'))

    # Use the plt.xticks function to custom labels
    plt.xticks(y_pos, bars, rotation=45, horizontalalignment='right')

    # Remove labels
    plt.tick_params(labelbottom='off')

    plt.title('Percent of Patients by Fetal Health Status')
    plt.savefig(os.path.join(VIS_DIR, 'targetdist.jpg'))
    plt.clf()

    # Create another figure
    plt.figure(figsize=(10, 6))

    # Scatter with normal examples
    plt.scatter(df.baseline_value[df.fetal_health == 1],
                df.accelerations[df.fetal_health == 1],
                c="yellowgreen")
    # Scatter with suspect examples
    plt.scatter(df.baseline_value[df.fetal_health == 2],
                df.accelerations[df.fetal_health == 2],
                c="gold")

    # Scatter with pathologic examples
    plt.scatter(df.baseline_value[df.fetal_health == 3],
                df.accelerations[df.fetal_health == 3],
                c="firebrick")

    # Add some helpful info
    plt.title("Fetal Health Status in function of Baseline FHR and Accelerations")
    plt.ylabel("Accelerations")
    plt.xlabel("Baseline FHR")
    plt.legend(["Normal", "Suspect", "Pathologic"])
    plt.savefig(os.path.join(VIS_DIR, 'base_acc.jpg'))
    plt.clf()

    pd.crosstab(df.fetal_health, df.prolongued_decelerations).plot(kind="bar",
                                                                   figsize=(10, 6),
                                                                   color=["yellowgreen", "gold", "firebrick"])

    # Add some communication
    plt.title("Fetal Health Status in function of Prolonged Decelerations")
    plt.ylabel("Histogram Tendency")
    plt.xlabel("Amount")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(VIS_DIR, 'pro_dec.jpg'))
    plt.clf()

    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(15, 10))
    ax = sns.heatmap(corr_matrix,
                     annot=True,
                     linewidths=0.5,
                     fmt=".2f",
                     cmap="YlGnBu")

    plt.savefig(os.path.join(VIS_DIR, 'corr.jpg'))
    plt.clf()


#dataAnalysis()