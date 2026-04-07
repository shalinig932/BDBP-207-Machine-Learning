#this file has all the functions in detail, helpful in revision.
import numpy as np
import pandas as pd
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as snp
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data():
    df =pd.read_csv('iris.csv')
    X = df.iloc[:,:-1]
    y= df.iloc[:,-1]
    return X,y

def EDA(X):
    print(X.shape)
    print(X.info())#shows col names,non-null counts and data types
    print(X.describe(include='all'))#this gives all the categorical and numerical data
    print(X.describe().T)#this gives the transpose of X
    print(X.head())
    print(X.tail())
    print(X.isnull().sum())#converts the dataframe into true/false . if there are missing values then
    #sum==counts it returns as 0=true,false=1
    print(X.corr())
    print('missing % of null is:',X.isnull().sum() * 100/len(X))#missing values
    plt.figure(figsize=(10,8))
    snp.heatmap(X.corr(),annot=True,fmt='.2f')
    plt.show()

#there are two simple steps to this first is to fill the misssing values of that features by mean,mode n median,
#then to scale the features.
def preprocessing(X):
   num_features=Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
   cat_features=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder())])
   return num_features,cat_features



def main():
    X,y=load_data()
    EDA(X)
    preprocessing(X)

if __name__ == '__main__':
    main()

