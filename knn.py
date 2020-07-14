from sklearn import datasets,neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
data=pd.read_csv('car.data')
# print(data.head())
x=data[['buy','maint','safety']].values
y=data[['class']]
# print(x[0])
# print(x,y )
le=LabelEncoder()
#conversion of x
for i in range (len(x[1])):
    x[:,i]=le.fit_transform(x[:,i])
# print(x)
#conversion of y
label_map={'unacc':0,'acc':1,'good':2,'vgood':3}
y['class']=y['class'].map(label_map)
y=np.array(y)
# print(y)
#create a knn model
knn=neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform')
#train the model

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.1)
knn.fit(xtrain,ytrain)
#predicting the test dataset
prediction=knn.predict(xtest)

#calculating accuracy
accuracy=metrics.accuracy_score(ytest,prediction)
print('prediction',prediction)
print('accuracy',accuracy)
#checking for an actual value
a=500 #this value can be changed
print('actual value:',y[a])
print('predicted value:',knn.predict(x)[a])




# save the model to disk
# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))
#opening a model
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)


#visualing a the data
# x=np.linspace(1,100,50)
# y=x*3+5
# rnd=np.random.RandomState(50)
# x=rnd.randint(1,10,size=(1,50))
# x=np.sort(x)
# print(x)
# y=x*x-2*x+3
# print(y)
# plt.scatter(x[:,0],y,c='r')
# plt.scatter(x[:,1],y,c='g')
# plt.scatter(x[:,2],y,c='b')
# plt.scatter(x[:,3],y,c='y')
# # plt.plot(y.T,x.T,c='g')
# plt.show()

