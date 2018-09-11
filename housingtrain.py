import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

#load data from csv file
data = pd.read_csv('Housing.csv')
X = data[['lotsize','bedrooms','bathrms','stories']]
Y = data['price']

#split data into train and test.
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

#create regression model and fit training data
reg = LinearRegression()
lr = reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

#Theta coefficeients
print("Coeffecients= ",reg.coef_)
#Intercepts
print("Intercept= ",reg.intercept_)

#test prediction score
from sklearn.metrics import r2_score
r = r2_score(y_test,y_pred)
print("R2_Score= ",r)

fname = 'model.pkl'
model = open(fname, 'wb')
pickle.dump(lr,model)
pickle.dump(reg,model)
print("Trained Model Created")
model.close()
