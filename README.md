# Ex-05-Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
# STEP 1 :
Use the standard libraries in python for finding linear regression.

# STEP 2 :
Set variables for assigning dataset values.

# STEP 3 :
Import linear regression from sklearn.

# STEP 4:
Predict the values of array.

# STEP 5:
Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

# STEP 6 :
Obtain the graph.

## Program:
```
Developed by:Vikash s 
RegisterNumber:212222240115
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

df=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=df[:,[0,1]]
y=df[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  return J

def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y) / X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(X_train,y),
                        method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min() - 1,X[:,0].max()+1
  y_min,y_max=X[:,1].min() - 1,X[:,0].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1))

  X_plot = np.c_[xx.ravel(),yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label='admitted')
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label='NOT admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob = sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```
## Output:
# Array Value of x :
![image](https://github.com/Yogabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118899387/3317bcb2-390d-4c19-aacc-4a55b97153c0)

# Array Value of y :
![image](https://github.com/Yogabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118899387/b855d8c3-c45c-45a7-94cf-df16ee7de5fd)

# Exam 1 - score graph :
![image](https://github.com/Yogabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118899387/1cb89d3d-5dba-4507-8878-2885d022404c)

# Sigmoid function graph :
![image](https://github.com/Yogabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118899387/0c9c3a77-74cc-478a-a7d1-071738edc59b)

# X_train_grad value :
![image](https://github.com/Yogabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118899387/e3463c83-de90-4d51-8af7-d36162072e04)

# Y_train_grad value :
![image](https://github.com/Yogabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118899387/ec8b364c-aa07-46af-b982-14f9866ef2cc)

# Print res.x :
![image](https://github.com/Yogabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118899387/3aa9e70a-6ff2-438e-ae9b-07bc57c67996)

# Decision boundary - graph for exam score :
![image](https://github.com/Yogabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118899387/a3c98b85-5d1a-4f5d-a95d-299fb86370ca)

# Proability value :
![image](https://github.com/Yogabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118899387/d829bc2c-af95-4016-9573-64154cb49ef0)

# Prediction value of mean :
![image](https://github.com/Yogabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118899387/33d5000c-494f-4dcc-a6fb-caa749e3c8db)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.



