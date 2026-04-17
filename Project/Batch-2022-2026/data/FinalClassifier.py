import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import joblib
#Using this class to make my print statement look bold
class color:
    BOLD = '\033[1m'

model_metrics={}
def createModel():
    df = pd.read_csv("housing.csv")
    df = df.sample(frac=1)

    # Replace ocean proximity with values to help our model
    ocean_proximity = {a: b for b, a in enumerate(df['ocean_proximity'].unique())}
    df.replace(ocean_proximity, inplace=True)

    # The feature 'total_bedrooms' has NaN values so I will replace them with the mean of the feature
    df = df.apply(lambda x: x.fillna(x.mean()))

    df[df['median_house_value'] > 450000]['median_house_value'].value_counts().head()
    df = df.loc[df['median_house_value'] < 500001, :]
    df = df[df['population'] < 25000]

    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)
    lg = lgb.LGBMRegressor()
    lg.fit(X_train, y_train)

    # Randomized Search CV
    n_estimators = [int(x) for x in np.linspace(start=5, stop=350, num=2)]

    max_depth = [int(x) for x in np.linspace(3, 450, num=3)]

    max_depth.append(None)

    grid = {'n_estimators': n_estimators, 'learning_rate': np.linspace(0, 0.5, 5)}

    print(grid)

    rscv = RandomizedSearchCV(estimator=lg,
                              param_distributions=grid,
                              n_iter=15,
                              scoring='r2',
                              cv=10,
                              verbose=0,
                              random_state=42,
                              n_jobs=-1,
                              return_train_score=True)

    rscv.fit(X_train, y_train)

    # summarize result
    print(rscv.best_params_)
    print(rscv.score(X_test, y_test))

    # Predict on test data
    y_pred = rscv.predict(X_test)

    # Metrics Calculation
    r2 = round(metrics.r2_score(y_test, y_pred), 2)
    Adj_r2 = round(1 - (1 - r2) * (9 - 1) / (9 - 1 - 1), 2)

    # Display results
    print(color.BOLD + '\nR2 score is ', r2)
    model_metrics['R2_Score']=r2
    print(color.BOLD + '\nAdjusted R2 score is ', Adj_r2)
    model_metrics['Adj_R2_Score'] = Adj_r2
    print(color.BOLD + '\nMean Absolute Error is', round(metrics.mean_absolute_error(y_test, rscv.predict(X_test)), 2))
    model_metrics['mae'] = round(metrics.mean_absolute_error(y_test, rscv.predict(X_test)), 2)
    print(color.BOLD + '\nMean Squared Error is', round(metrics.mean_squared_error(y_test, rscv.predict(X_test)), 2))
    model_metrics['mse']=round(metrics.mean_squared_error(y_test, rscv.predict(X_test)), 2)
    print(color.BOLD + '\nRoot Mean Squared Error is',
          round(np.sqrt(mean_squared_error(y_test, rscv.predict(X_test))), 2))
    model_metrics['rmse']=round(np.sqrt(mean_squared_error(y_test, rscv.predict(X_test))), 2)
    # Exporting the model using joblib library
    joblib.dump(lg, "California_Model.pkl")

    msg="Model created using HyperParameter Tuning of Light Gradient Boosting Regressor"
    return msg, model_metrics

#createModel()
