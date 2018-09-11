import numpy as np
import pandas as pd
import pickle

#load data from csv file
data = pd.read_csv('Housing.csv')
X = data[['lotsize','bedrooms','bathrms','stories']]
Y = data['price']

#load model
fname = 'model.pkl'
model = open(fname, 'rb')
lr = pickle.load(model)
reg = pickle.load(model)
print("Cluster model loaded")

#details of house as input
print('Enter House details:','\n')
size = input('Area:')
bed = input('Number of Bedrooms: ')
bath = input('Number of Bathroom:')
stor = input('Floors: ')
print('\n')

#prediction
z = [size,bed,bath,stor]
z = list(map(int, z))
z = [z]
y_pred = lr.predict(z)
print('Predcted price =',y_pred[0])
