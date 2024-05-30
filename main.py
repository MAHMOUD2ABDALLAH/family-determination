


# PREPARATION 
import sklearn
import numpy as np
import pandas as pd
#cleaning data nal values
from sklearn.impute import SimpleImputer
#scaling and normalizing data 
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
#feature selection from data
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import selectpercentile
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import selectKbest
from sklearn.feature_selection import chi2 



#main
seg = pd.read_csv("G:\\classification\\train.csv")
print(seg.shape())

# Encoding categorical data
encoding = LabelEncoder()
seg['Gender'] = encoding.fit_transform(seg['Gender'])
seg['Ever_Married'] = encoding.fit_transform(seg['Ever_Married'])
seg['Graduated'] = encoding.fit_transform(seg['Graduated'])
seg['Profession'] = encoding.fit_transform(seg['Profession'])
seg['Spending_Score'] = encoding.fit_transform(seg['Spending_Score'])
seg['Var_1'] = encoding.fit_transform(seg['Var_1'])
seg['Segmentation'] = encoding.fit_transform(seg['Segmentation'])

# preprocessing scaling
scale = MinMaxScaler(copy=True, feature_range=(0, 1))
a = np.array(seg['ID'], dtype=int64)
seg['ID'] = scale.fit_transform(a.reshape(-1, 1))
scale = StandardScaler(copy=True, feature_range=(0, 1))
a = np.array(seg['ID'], dtype=int64)
seg['ID'] = scale.fit_transform(a.reshape(-1, 1))

# Data cleaning
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
x1 = np.array(seg['Ever_Married'], dtype=int64)
seg['Ever_Married'] = imp.fit_transform(x1.reshape(-1, 1))
x2 = np.array(seg['Graduated'], dtype=int64)
seg['Graduated'] = imp.fit_transform(x2.reshape(-1, 1))
x3 = np.array(seg['Profession'], dtype=int64)
seg['Profession'] = imp.fit_transform(x3.reshape(-1, 1))
x4 = np.array(seg['Work_Experience'], dtype=int64)
seg['Work_Experience'] = imp.fit_transform(x4.reshape(-1, 1))
x5 = np.array(seg['Family_Size'], dtype=int64)
seg['Family_Size'] = imp.fit_transform(x5.reshape(-1, 1))
x6 = np.array(seg['Var_1'], dtype=int64)
seg['Var_1'] = imp.fit_transform(x6.reshape(-1, 1))

#feature selection methods
seg = Selectpercentile(select_func = chi2 , percentile = 40)
seg = GenericUnivariateSelect(select_func = chi2 ,mode = 'k_best' , param = 4)
seg = SelectKBest(select_func = chi2 ,k = 5)
seg = SelectFromModel(estimator = LinearRegression(), max_features = none)
Selected = seg.fit_transform(X, Y)
print(Selected.shape)
print(seg.get_support())







