# !/usr/bin/env python3
# coding: utf-8

# # Ajuste lineal con una variable independiente

# En este caso, las variables se corresponden con columnas de un dataframe de pandas.

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston


# In[7]:


# Change to a specific directory
#os.chdir("C:/Users/diego/Desktop/Python/Python_Notebooks")


# Fit a linear regression:
# - X: 'lower status of population' ('lstat')
# - Y: 'Median value of owner occupied homes' ('medv')

# In[8]:


# Read the CSV file
df = pd.read_csv("Boston.csv", encoding = "ISO-8859-1")

# Alternativa: Cargamos los datos como Bunch object.
# boston = load_boston()
# print (boston)

print (df.head(7))
print (df.shape) # (506, 15)


# In[9]:


# Select the feature variable (variable independiente)
X = df['lstat']
type(X) # pandas.core.series.Series
# print(X.shape) # (506,)
# Nótese que X es un array 1D.
# Para el ajuste debe ser un array 2D, con una sola columna, es decir (506,1).


# In[10]:


# Select the target (variable dependiente)
y = df['medv']
type (y) # pandas.core.series.Series
# print(y.shape) # (506,)


# In[11]:


# Split into train and test sets (75:25)
# Empleamos la función train_test_split del paquete scikit learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)
# print(X_train)
print(X_train.shape) # (379,) 

# Para emplear el ajuste lineal, X debe ser un array 2D.
# Si no, devuelve el error: "Expected 2D array, got 1D array instead".
# La función 'reshape()' permite redimensionar una matriz.
# Con el segundo índice (1) le indicamos que tiene que tener una columna.
# Con el primer índice (-1) le decimos que calcule las filas necesarias.
X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)
# print (X_train.shape) # (379, 1)
# print (y_train.shape) # (379,)


# In[12]:


# Fit a linear model
linreg = LinearRegression().fit(X_train, y_train)


# In[13]:


# Print the training and test R squared score
print('R-squared score (training): {:.3f}'.format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(linreg.score(X_test, y_test)))


# In[14]:


print("Pendiente: " + str(linreg.coef_))
print("Término independiente: " + str(linreg.intercept_))


# In[19]:


# Representamos los datos:
plt.figure(figsize=(15,8))
fig1 = plt.scatter(X_train,y_train)
fig1 = plt.title("Median value of homes ($1000) vs Lower status (%)")
fig1 = plt.xlabel("Lower status (%)")
fig1 = plt.ylabel("Median value of owned homes ($1000)")

# Incluímos la recta de ajuste en la figura.
# Create a range of points. Compute yhat=coeff1*x + intercept and plot
x = np.linspace(0,40,20) # 20 puntos equiespaciados entre 0 y 40.
fig1 = plt.plot(x, linreg.coef_ * x + linreg.intercept_, color='red')

#plt.show()
plt.savefig('Grafica1.png')


# In[20]:


# Hacer predicciones sobre el conjunto de test:
y_pred = linreg.predict(X_test)
# Mean square error:
print ("Mean squared error: %.2f" % mean_squared_error (y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))


# In[21]:


# Representamos las predicciones:
plt.figure(figsize=(15,8))
fig2 = plt.plot(y_test, y_pred, linestyle='none', markerfacecolor='blue', 
                 marker="o", markeredgecolor="black", markersize=3)
fig2 = plt.title("Predicciones")
fig2 = plt.xlabel("Y_test")
fig2 = plt.ylabel("Y_pred")
#plt.show()
plt.savefig('Grafica.png')

