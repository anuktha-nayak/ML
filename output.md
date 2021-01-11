```python
The execise has been coded in Python which have similar libraries as in R and the same logic follows. The data is first processed following which data is split into train and test. XGB is used along with grid search to train the model. The accuracy is almost 100% which shows how powerful XGB is. The model is then used to predict on the test data.
```


```python
#Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
```


```python
#Reading the files for train and test
df=pd.read_csv('pml-training.csv')
df_test=pd.read_csv('pml-testing.csv')
df.drop(columns='Unnamed: 0',inplace=True)
df_test.drop(columns='Unnamed: 0',inplace=True)
```

    /opt/conda/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3146: DtypeWarning: Columns (11,14,19,22,25,70,73,86,87,89,90,94,97,100) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
#Data Pre-processing has been carried out. Specifically, columns with more than 19,000 null values
# are identified and removed
missing=[]
for i in df.columns:
    if (df[i].isnull().sum() > 19000):
        #print (i,"has",df[i].isnull().sum())
        missing.append(i)

df.drop(columns=missing,inplace=True)
df_test.drop(columns=missing,inplace=True)
```


```python
#Categorical columns and time columns which has no info on target are removed after examining data
drop=['user_name','raw_timestamp_part_1','raw_timestamp_part_2','cvtd_timestamp','new_window']
target=['classe']

df.drop(columns=drop,inplace=True)
df_test.drop(columns=drop,inplace=True)
```


```python
#The feature variables and target are separated
y=df[target]
x=df.drop(columns=target)
```


```python
#Train and test are split in 70:30 ratio
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3)
```


```python
#XGB Classifier libraries are imported to train the model
import sklearn
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV  
```


```python
#Initialising the XGB model with default parameters
xg=xgb.XGBClassifier(random_state=1)
xg.fit(x_train, y_train['classe'])
```




    XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints=None,
                  learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                  min_child_weight=1, missing=nan, monotone_constraints=None,
                  n_estimators=100, n_jobs=0, num_parallel_tree=1,
                  objective='multi:softprob', random_state=1, reg_alpha=0,
                  reg_lambda=1, scale_pos_weight=None, subsample=1,
                  tree_method=None, validate_parameters=False, verbosity=None)




```python
#Using the default parameters for the model
xgb_clf = XGBClassifier(learning_rate =0.300000012, n_estimators=100, max_depth=6,
 min_child_weight=1, gamma=0, subsample=1, colsample_bytree=1,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
```


```python
#Hyper-Parameter-Tuning: Defining standard set of ranges for the parameters
#Max depth is the max# nodes allowed from root to farthest leaf. If too high, leads to overfitting
#min_child_weight is min weight required to create a new node
#Subsample is the fraction of observations to subsample at each stage
#reg_alpha & reg_lambda is to reduce overfitting used in line with lasso regression
#Learning rate is slope and should be low to reduce overfitting
#n_estimators is the number of decision trees to be made
#gamma is used for regularization to prevent overfitting

param_test = {
 'max_depth':range(3,10,2),'min_child_weight':range(1,6,2),'subsample':[i/10.0 for i in range(6,11)],
    'colsample_bytree':[i/10.0 for i in range(4,11)],'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
    'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100],'learning_rate':[0.0001,0.001,0.01,0.1,0.2,0.3],
    'scale_pos_weight':[1,2,3,4,5,6,7,8,9,10,11],'n_estimators':[100,300,500,700,900],
    'gamma':[i/10.0 for i in range(0,5)]
}
```


```python
#Running a grid search that breaks the train set into 5 folds and tries every combination of parameters
#and scores on accuracy of classification
keys=list(param_test.keys())

for i in range(0,len(keys)):
    param={keys[i]:param_test[keys[i]]}
    gsearch = GridSearchCV(estimator = xgb_clf,param_grid = param, scoring='accuracy',iid=False, cv=5)
    gsearch.fit(x_train,y_train['classe'])
    print(gsearch.best_params_, gsearch.best_score_)
    param=gsearch.best_params_
    xgb_clf.set_params(**param)
```

    {'max_depth': 5} 0.998980546968847
    {'min_child_weight': 5} 0.9991262134699657
    {'subsample': 1.0} 0.9991262134699657
    {'colsample_bytree': 0.7} 0.9991989671214073
    {'reg_alpha': 0.1} 0.999199020168182
    {'reg_lambda': 1} 0.999199020168182
    {'learning_rate': 0.3} 0.999199020168182
    {'scale_pos_weight': 1} 0.999199020168182
    {'n_estimators': 100} 0.999199020168182
    {'gamma': 0.0} 0.999199020168182



```python
#Fitting the final model on the train
model= xgb_clf
model.fit(x_train,y_train['classe'])
```




    XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=0.7, gamma=0.0, gpu_id=-1,
                  importance_type='gain', interaction_constraints=None,
                  learning_rate=0.3, max_delta_step=0, max_depth=5,
                  min_child_weight=5, missing=nan, monotone_constraints=None,
                  n_estimators=100, n_jobs=4, nthread=4, num_parallel_tree=1,
                  objective='multi:softprob', random_state=27, reg_alpha=0.1,
                  reg_lambda=1, scale_pos_weight=1, seed=27, subsample=1.0,
                  tree_method=None, validate_parameters=False, verbosity=None)




```python
#Using the model to predict on the test and train
y_pred_train=model.predict(x_train)
y_pred_test=model.predict(x_test)
```


```python
#Accuacy of test set is almost 100%
print('Accuracy of XGB classifier on test set: {:.2f}'.format(model.score(x_test, y_test)))
```

    Accuracy of XGB classifier on test set: 1.00



```python
#Accuracy on test set is also close to 100% thus the model has not been overfitted
print('Accuracy of XGB classifier on test set: {:.2f}'.format(model.score(x_train, y_train)))
```

    Accuracy of XGB classifier on test set: 1.00



```python
#printing the classification report for test and train data sets
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_train['classe'],y_pred_train))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test['classe'],y_pred_test))
```

                  precision    recall  f1-score   support
    
               A       1.00      1.00      1.00      3908
               B       1.00      1.00      1.00      2630
               C       1.00      1.00      1.00      2399
               D       1.00      1.00      1.00      2262
               E       1.00      1.00      1.00      2536
    
        accuracy                           1.00     13735
       macro avg       1.00      1.00      1.00     13735
    weighted avg       1.00      1.00      1.00     13735
    
                  precision    recall  f1-score   support
    
               A       1.00      1.00      1.00      1672
               B       1.00      1.00      1.00      1167
               C       1.00      1.00      1.00      1023
               D       1.00      1.00      1.00       954
               E       1.00      1.00      1.00      1071
    
        accuracy                           1.00      5887
       macro avg       1.00      1.00      1.00      5887
    weighted avg       1.00      1.00      1.00      5887
    



```python
#printing the confusion matrix on test. Only four has been wrongly classified
confusion = pd.crosstab(y_test['classe'], y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(confusion)
```

    Predicted     A     B     C    D     E   All
    Actual                                      
    A          1672     0     0    0     0  1672
    B             1  1166     0    0     0  1167
    C             0     0  1023    0     0  1023
    D             0     0     0  954     0   954
    E             0     1     0    3  1067  1071
    All        1673  1167  1023  957  1067  5887



```python
#Checking if the feature variables are the same for pml-testing and pml-training sets
set(df_test.columns)-set(df.columns)
set(df.columns)-set(df_test.columns)
df_test.drop(columns='problem_id',inplace=True)
```


```python
#Predicting the pml-testing data 
y_pred_val=model.predict(df_test)
y_pred_val
```




    array(['B', 'A', 'B', 'A', 'A', 'E', 'D', 'B', 'A', 'A', 'B', 'C', 'B',
           'A', 'E', 'E', 'A', 'B', 'B', 'B'], dtype=object)


