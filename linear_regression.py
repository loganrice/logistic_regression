import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

ficoScores = loansData['FICO.Range'].map(lambda scores: scores.split("-"))
ficoScores = ficoScores.map(lambda scores: [int(score) for score in scores])
ficoScores = ficoScores.map(lambda scores: scores[0])
loansData['FICO.Score'] = ficoScores

loanLength = loansData['Loan.Length']
loanLength = loanLength.map(lambda length: length.rstrip(" months"))
loanLength = loanLength.map(lambda length: int(length))
loansData['Loan.Length'] = loanLength

interestRates = loansData['Interest.Rate']
interestRates = interestRates.map(lambda rate: rate.rstrip("%"))
interestRates = interestRates.map(lambda rate: float(rate) / 100)
loansData['Interest.Rate'] = interestRates

loanAmount = loansData['Amount.Requested']
# plt.figure()
p = loansData['FICO.Score'].hist()
# plt.show()
# plt.clf()

# plt.figure()
# pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10))
pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')
# plt.show()

# The dependent variable
y = np.matrix(interestRates).transpose()
# The independent variables shaped as columns
x1 = np.matrix(ficoScores).transpose()
x2 = np.matrix(loanAmount).transpose()
x = np.column_stack([x1,x2])
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()
print f.summary() 
