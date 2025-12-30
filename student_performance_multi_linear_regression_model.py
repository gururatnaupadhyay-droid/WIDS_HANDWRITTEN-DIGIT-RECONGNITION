
'''
This file contains the Worksheet from the second week's code. 
We had to create a multi linear regression model to model the expected outcome/marks of a test of students given some parameters.
These paramters were dependent variables and we had to figure out the weights and biases base on training from a give data set
The techinques used was gradient descent.
We also had to compare our answers with that of Scikit learn's inbuilt function.

ADDITIONALLY FOR THE SAKE OF LEARNING I ALSO CALCULATED THE WEIGHTS AND BIASES IN A NON ITERATIVE MANNER.
THIS USES A ONE SHOT FORMULA THAT CAN BE PROVED BY DOING A LITTLE PEN AND PAPER WORKING OUT!
THIS FORMULA WAS GIVEN BY NEWTON.

I HAD AN IDEA AND TRIED TO APPLY AN INCORRECT FORMULA TO GIVE THE BIASES AND THE WEIGHTS IN A SINGLE ITERATION.
AFTER TESTING AND FINDING HORRIBLE ACCURACY, I CHATGPTED THE THING AND THAT LEAD TO ME THE FORMULA WHICH CAN BE PROVED EASILY AND USED.
THIS IS GOOD FOR SMALLER NUMBER OF PARAMETERS AND WORKS OK FOR MINIMISING THE SUM OF SQUARES OF DIFFERENCES ONLY
AND NOT IF THE NUMBER OF PARAMETERS IS LARGE OR THE LOSS FUNCTION WERE SOMETHING ELSE(SAY, POLYNOMIAL)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
import math

df = pd.read_csv('Student_Performance.csv')
dataframe=df
# Encode categorical into a binary yes or no
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({
    'Yes': 1,
    'No': 0
})


cols_to_scale = ['Hours Studied', 'Sleep Hours','Previous Scores','Sample Question Papers Practiced']

for x in cols_to_scale:
  df[x] = (df[x] - df[x].mean()) / df[x].std()      #Z Score Standardisation
test=df[8000::1]                                # test is the dataframe that stores the data containing the test cases

df=df[:8000:1]                                # df contains the data for testing
X = df.drop('Performance Index', axis=1).values     #training data
y = df['Performance Index'].values.reshape(-1, 1) # training answers   (Reshape y to be a column vector)

n_samples = X.shape[0]
X = np.c_[np.ones((n_samples, 1)), X]
'''
cols=[]
for i in range (6):
  a=0
  for b in range (8000):
    a+=X[b][i]**2
  cols.append(a)

cols=np.array(cols)
cols=cols.reshape(-1,1)
'''

n_features = X.shape[1]
beta = np.zeros((n_features, 1))
def gradient_descent(X, y, beta, lr=0.01, epochs=1000):
    n = len(y)
    losses = []

    for a in range(epochs):
        y_pred = X @ beta
        error = y_pred - y

        gradient = (2/n) * (X.T @ error)
        beta -= lr * gradient

        loss = np.mean(error ** 2)
        losses.append(loss)

    return beta, losses

b, losses = gradient_descent(X, y, beta, lr=0.01, epochs=900)

beta=b                #<---- THIS IS THE LIST OF THE WEIGHTS CALCULATED BY GRADIENT DESCENT


beta2 = np.linalg.inv(X.T @ X) @ X.T @ y #  <------THIS IS THE LIST OF THE WEIGHTS CALCULATED BY NEWTON'S FORMULA

output=np.c_[y,X@beta,X@beta2]
np.savetxt('output_file.csv',output, delimiter=',')


def evaluate_model(y_actual, y_pred):
    mse = np.mean((y_actual - y_pred)**2)
    ss_res = np.sum((y_actual - y_pred)**2)
    ss_tot = np.sum((y_actual - np.mean(y_actual))**2)
    r2 = 1 - ss_res/ss_tot
    return mse, r2

y_actual = y # Use the original y values
y_pred = X @ b # Calculate predictions using the final beta

mse, r2 = evaluate_model(y, X@beta)
mse2,r22=evaluate_model(y, X@beta2)

print(1-mse2/mse)
print (r2<r22)
print(cols)
print(output)


# TESTING PHASE 

ye=test['Performance Index'].values.reshape(-1, 1)      #training answers
te=test.drop('Performance Index', axis=1).values    #testing data
te=np.c_[np.ones((len(te), 1)), te]
exp=te@beta
d=abs(exp-ye)/ye
print(d.mean())

exp2=te@beta2
d2=abs(exp2-ye)/ye
print(d2.mean())




#COMPARING WITH SCIKIT LEARN

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X, y)

y_pred_sklearn = model.predict(te)

mse_sklearn = mean_squared_error(ye, y_pred_sklearn)
r2_sklearn  = r2_score(ye, y_pred_sklearn)

print(mse_sklearn,mse)
print(r2_sklearn,r2)
