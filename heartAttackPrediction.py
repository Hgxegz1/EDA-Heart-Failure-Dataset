'''This is a prediction analysis on if a patient will have a heart attack or not
given the dataset'''

'''heart attack dataset details
1.)age=age in years
2.)sex=sex(0 = female, 1 = male)
3.)cp=chestpain type(1=typical angina,2=atypical angina,3=non-anginal pain,0=asymptomatic)
4.)trtbps=resting blood pressure(in mm Hg on admission to the hospiatal)
5.)chol=esrum cholestoral in mg/dl
6.)fbs=fasting blood sugar > 120 mg/dl (0=false, 1=true)
7.)restecg=resting electrocardiographic results(0=normal,1=hypertrophy,2=having ST-T abnorm)
8.)thalachh=maximum heart rate achieved
9.)exng=exercise induced angina(0 = no, 1 = yes)
10.)oldpeak=ST depression induced by exercise relative to rest
11.)slp=the slope of the peak exercise ST segment(0=downsloping, 1=flat, 2=upsloping)
12.)caa=number of major vessels (0-4) colored by flourosopy
13.)thall=thallium stress test(1=fixed defect, 2=reversable defect, 3=normal)
14.)output= 0=less chance of heart attack, 1=more chance of heart attack '''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform
from scipy.stats import loguniform
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from catboost import CatBoostClassifier

print("Everyhing imported correctly")

df = pd.read_csv("heart.csv")
print(df.head())
