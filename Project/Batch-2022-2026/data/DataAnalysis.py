import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dataAnalysis():
    df = pd.read_csv("housing.csv")
    df = df.sample(frac=1)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='ocean_proximity')
    plt.savefig('static/vis/op.jpg')
    plt.clf()

    # Replace ocean proximity with values to help our model
    ocean_proximity = {a: b for b, a in enumerate(df['ocean_proximity'].unique())}
    df.replace(ocean_proximity, inplace=True)

    # The feature 'total_bedrooms' has NaN values so I will replace them with the mean of the feature
    df = df.apply(lambda x: x.fillna(x.mean()))

    df[df['median_house_value'] > 450000]['median_house_value'].value_counts().head()
    df = df.loc[df['median_house_value'] < 500001, :]
    df = df[df['population'] < 25000]

    plt.figure(figsize=(11, 7))
    sns.heatmap(cbar=False, annot=True, data=df.corr() * 100, cmap='coolwarm')
    plt.title('% Corelation Matrix')
    plt.savefig('static/vis/cor.jpg')
    plt.clf()

    plt.figure(figsize=(10, 6))
    sns.stripplot(data=df, x='ocean_proximity', y='median_house_value', jitter=0.3)
    plt.savefig('static/vis/oc_mhv.jpg')
    plt.clf()


    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']

#dataAnaysis()

