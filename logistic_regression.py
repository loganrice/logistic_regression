import pandas as pd
import statsmodels.api as sm
import math

df = pd.read_csv('loansdata_clean.csv')
interestRates = df['Interest.Rate']
df['IR_TF']  = interestRates.map(lambda rate: 0 if rate < 0.12 else 1)
df['Intercept'] = 1.0
ind_vars = ['FICO.Score', 'Amount.Requested', 'Intercept']

logit = sm.Logit(df['IR_TF'], df[ind_vars])

result = logit.fit()
coeff = result.params
fico_factor = coeff[0]
loan_factor = coeff[1]
intercept = coeff[2]

def linear_function(ficoScore, loanAmount):
    interest_rate = ( -1 * intercept) + ( -1 * fico_factor * ficoScore) + (-1 * loan_factor * loanAmount)
    return interest_rate

def logistic_function(ficoScore, loanAmount):
    return 1 / (intercept + math.e ** (1 + (fico_factor * ficoScore) + (loan_factor * loanAmount)))
    


print logistic_function(750, 10000)


