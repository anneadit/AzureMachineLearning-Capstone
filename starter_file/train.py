from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace
from azureml.core import Dataset, Datastore


run = Run.get_context()
ws = run.experiment.workspace


ds = Dataset.get_by_name(ws, name='titanic-survival')

def clean_data(data):

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe()

    #x_df.drop("cabin", inplace = True, axis = 1)
    
    #x_df['embarked'].fillna(x_df['embarked'].mode(), inplace = True)
    embarked = pd.get_dummies(x_df.embarked, prefix="embarked")
    x_df.drop("embarked", inplace=True, axis=1)
    x_df = x_df.join(embarked)

    #x_df['cabin'].fillna(x_df['cabin'].mode(), inplace = True)
    cabin = pd.get_dummies(x_df.cabin, prefix="cabin")
    x_df.drop("cabin", inplace=True, axis=1)
    x_df = x_df.join(cabin)

    x_df["sex"] = x_df.sex.apply(lambda s: 1 if s == "female" else 0)
    
    x_df['sex'].fillna(x_df['sex'].mode(), inplace = True)
    x_df['pclass'].fillna(x_df['pclass'].mode(), inplace = True)
    x_df['sibsp'].fillna(x_df['sibsp'].mode(), inplace = True)
    x_df['parch'].fillna(x_df['parch'].mode(), inplace = True)
    x_df['survived'].fillna(x_df['survived'].mode(), inplace = True)
  
    x_df['fare'].fillna(x_df['fare'].mean(), inplace = True)
    x_df['age'].fillna(x_df['age'].mean(), inplace = True)

    x_df.fillna(method = 'ffill', inplace = True)

    y_df = x_df.pop("survived")

    scaler = MinMaxScaler()
    x_df = scaler.fit_transform(x_df)

    return x_df, y_df
    
x, y = clean_data(ds)

# Split data into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('./outputs', exist_ok=True)
    f = open('./outputs/model.pkl', 'wb')
    pickle.dump(model, f)
    f.close()

if __name__ == '__main__':
    main()