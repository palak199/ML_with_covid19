#here i am trying to analyse day wise growth in covid cases in state of punjab




import pandas as pd;         #is used to read and write tabular data
import numpy as np;          #to convert data from csv to array
import matplotlib.pyplot as plt    #to visualise graphs
from sklearn.linear_model import LinearRegression
data=pd.read_csv('statewisedata.csv')
#initially I tried to train the model with linear regression 
#the bad model
# X = data['sno'].values.reshape(-1,1)
# y = data['confirmed cases'].values.reshape(-1,1)
# reg = LinearRegression()
# reg.fit(X, y)
# print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))
# predictions = reg.predict(X)
# plt.scatter(X,y,c="black")
# plt.plot(data['sno'],predictions,c="red" )
# plt.show()
#the goooood model
X=np.array(data['sno'])
y=np.array(data['confirmed cases'])
#custom fit
p = np.polyfit(X, np.log(y), 1)
print(f'The equation of regression line is ln(y)={p[0]} * ln(x) + {p[1]}')
#semi log plot worked !
plt.plot(X, p[0] * X + p[1], 'r--', label='Regression line')
plt.scatter(X,np.log(y),label='day vs cases')
plt.title("covid 19")
plt.xlabel('day')
plt.ylabel('cases')
plt.legend()
plt.show()
