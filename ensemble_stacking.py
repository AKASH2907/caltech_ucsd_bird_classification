from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import math 
import xgboost as xgb
np.random.seed(2019)
from scipy.stats import skew
from scipy import stats

# import statsmodels
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt


X_train = np.load("feature_vec.npy")
Y_train = np.load("feature_labels.npy")

X_test = np.load("test_vec.npy")
Y_test = np.load("test_tr_labels.npy")

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Voting Classifier

model_1 = LogisticRegression(random_state=1)
model_2 = RandomForestClassifier(random_state=1, n_estimators=100)
# model_3 = SVC(probability=True, random_state=1)
# model_3 = LinearSVC(random_state=1)
model_4 = KNeighborsClassifier(n_neighbors=3)

model = VotingClassifier(estimators=[('lr', model_1), ('ls', model_2), ('knc', model_4)], voting='soft')

model.fit(X_train, Y_train)
acc = model.score(X_test, Y_test)
print(acc)
# 75.23

# tuned_parameters = {'n_estimators':[500],'n_jobs':[-1], 'max_features': [0.5,0.6,0.7,0.8,0.9,1.0], 
#                     'max_depth': [10,11,12,13,14],'min_samples_leaf':[1,10,100],'random_state':[0]} 

# clf = GridSearchCV(ExtraTreesClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
# clf.fit(X_train, Y_train)
# y_pred = clf.predict(X_test)
# score = accuracy_score(Y_test, y_pred)
# print(score)

model1 = DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()
model4 = RandomForestClassifier()
model5 = GradientBoostingClassifier()

model1.fit(X_train,Y_train)
model2.fit(X_train,Y_train)
model3.fit(X_train,Y_train)
model4.fit(X_train,Y_train)
model5.fit(X_train,Y_train)

pred1=model1.predict(X_test)
pred2=model2.predict(X_test)
pred3=model3.predict(X_test)
pred4=model4.predict(X_test)
pred5=model5.predict(X_test)

print(pred1, pred2, pred3, pred4, pred5)

def Stacking(model,train,y,test,n_fold,t=1):
    folds=StratifiedKFold(n_splits=n_fold,random_state=1)
    test_pred=np.empty((0,1),float)
    train_pred=np.empty((0,1),float)
    for train_indices,val_indices in folds.split(train,y.values):
        x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
        Y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]
        if t==1:
            model.fit(X=x_train,y=Y_train)
        else:
            model.train(x_train,Y_train)
        train_pred=np.append(train_pred,model.predict(x_val))
    test_pred=np.append(test_pred,model.predict(test))
    return test_pred.reshape(-1,1),train_pred

nfolds = 5

model1 = DecisionTreeClassifier()

test_pred1 ,train_pred1=Stacking(model=model1,n_fold=nfolds, train=X_train,test=X_test,y=y_train)

train_pred1=pd.DataFrame(train_pred1)
test_pred1=pd.DataFrame(test_pred1)

model2 = KNeighborsClassifier()

test_pred2 ,train_pred2=Stacking(model=model2,n_fold=nfolds,train=X_train,test=X_test,y=y_train)

train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)

model3 = RandomForestClassifier()

test_pred3 ,train_pred3=Stacking(model=model3,n_fold=nfolds,train=X_train,test=X_test,y=y_train)

train_pred3=pd.DataFrame(train_pred3)
test_pred3=pd.DataFrame(test_pred3)


model4 = GradientBoostingClassifier()

test_pred4 ,train_pred4=Stacking(model=model4,n_fold=nfolds,train=X_train,test=X_test,y=y_train)

train_pred4=pd.DataFrame(train_pred4)
test_pred4=pd.DataFrame(test_pred4)


df = pd.concat([train_pred1, train_pred2, train_pred3, train_pred4], axis=1)
df_test = pd.concat([test_pred1, test_pred2, test_pred3, test_pred4], axis=1)

model = LogisticRegression(random_state=1)
model.fit(df,y_train)
#y_test = model.predict(df_test)

print("HiperModelo LogisticRegression",model.score(df_test, y_test))