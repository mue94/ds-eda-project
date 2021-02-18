import pandas as pd #import necassary packages
import statsmodels.api as sms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

df = pd.read_csv('us_bank_wages/us_bank_wages.txt', delimiter="\t") #read the csv-file

df.drop('Unnamed: 0', axis = 1, inplace = True) #drop unnecassary index column

educ_dummies = pd.get_dummies(df['EDUC'], prefix='edu', drop_first=True) #create dummie-variables
gender_dummies = pd.get_dummies(df['GENDER'], prefix='gd', drop_first=True)
minority_dummies = pd.get_dummies(df['MINORITY'], prefix='mino', drop_first=True)
jobcat_dummies = pd.get_dummies(df['JOBCAT'], prefix='jcat', drop_first=True)

df = df.drop(['EDUC','GENDER','MINORITY','JOBCAT'], axis=1) #drop origin columns

df = pd.concat([df, educ_dummies, gender_dummies, minority_dummies, jobcat_dummies], axis=1) #add created dummie variables

#feature engineering
X = df[['SALBEGIN', 'edu_12', 'edu_14', 'edu_15', 'edu_18', 'edu_19', 'edu_20', 'edu_21', 'mino_1', 'gd_1', 'jcat_2', 'jcat_3']]
Y = df['SALARY']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #splitting the dataset

X_train = sms.add_constant(X_train) #adding the constant
X_test = sms.add_constant(X_test)

model = sms.OLS(y_train, X_train).fit() #training the model
print_model = model.summary()

predictions = model.predict(X_train) #evaluating the model
err_train = np.sqrt(mean_squared_error(y_train, predictions))
predictions_test = model.predict(X_test)
err_test = np.sqrt(mean_squared_error(y_test, predictions_test))

print(print_model) #get the test size
print ("-------------") 
print (f"RMSE on train data: {err_train}")
print (f"RMSE on test data: {err_test}")

with open ('model','wb') as f: #save the model
    pickle.dump(model,f)
    
#usable with: with open ('model','rb') as f:
             # model = pickle.load(f)
             # e. g.  float(model.predict([1,40000,0,0,0,1,0,0,0,1,0,0,0])
             # Output: 61759.65801460731