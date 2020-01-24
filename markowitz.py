# Data analysis library
import pandas as pd
import numpy as np

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# List stock tickers in portfolio
stockPort = ['AAPL', 'FB', 'GE', 'JPM', 'MSFT', 'V']

# Create empty dataframe of daily stock returns matrix
stockRetMat = pd.DataFrame(columns=stockPort)

# Function: create a column of a stock's daily returns
def stockReturnCol(ticker):
    ticker = ticker + '.csv'
    data = pd.read_csv(ticker, parse_dates=['Date']) #import historical data in pandas dataframe
    data = data.sort_values(by='Date')
    data.set_index('Date', inplace=True)

    data['Returns'] = data['Adj Close'].pct_change() #create a column of daily returns
    stockReturns = data['Returns'].dropna() #remove entries with N/A dat
    return stockReturns

# For each stock in the portfolio, generate a column of its daily returns and store it in the stock return matrix
for stock in stockPort:
    stockRetMat[stock] = stockReturnCol(stock)

# Generate a correlation matrix of the stock returns using the Pearson method
corrMatrix = stockRetMat.corr(method='pearson')

def randomPortfolios(numPortfolios, meanReturns, covMatrix, riskFreeRate):
    results = np.zeros((3,numPortfolios))
    weightsRecord = []
    for i in range(numPortfolios):
        weights = np.random.random_sample(len(stockPort))
        weights /= np.sum(weights)
        weightsRecord.append(weights)
        #portfolioStdDev, portfolioReturn = portfolioPerformance(weights, meanReturns, covMatrix)
        portfolioReturn = np.sum(meanReturns * weights) * 252
        portfolioStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252) 
        results[0,i] = portfolioStdDev
        results[1,i] = portfolioReturn
        results[2,i] = portfolioReturn / portfolioStdDev
    return results, weightsRecord

returns = stockRetMat #matrix, 250 daily returns for each stock
meanReturns = returns.mean() #array, 1 average return for each stock
covMatrix = returns.cov() #square matrix of each stock
numPortfolios = 1000
riskFreeRate = 0

results, weights = randomPortfolios(numPortfolios, meanReturns, covMatrix, riskFreeRate)

#Max Sharpe ratio
maxSharpeIndex = np.argmax(results[2])
stdSharpe = results[0,maxSharpeIndex]
returnSharpe = results[1,maxSharpeIndex]
print(stdSharpe)
print(returnSharpe)

maxSharpeArray = weights[maxSharpeIndex].round(4)
maxSharpeAlloc = pd.DataFrame(maxSharpeArray, index=stockPort, columns=['Allocation']).T
print(maxSharpeAlloc)
print(type(maxSharpeAlloc))
maxSharpeAlloc = maxSharpeAlloc.T.sort_values(by=['Allocation'], ascending=False)
print(maxSharpeAlloc)
print(type(maxSharpeAlloc))

maxSharpeAlloc.plot.bar()

#Global minimum volatility
gmvIndex =np.argmax(results[0])
stdGmv = results[0,gmvIndex]
returnGmv = results[1,gmvIndex]
gmvArray = weights[gmvIndex].round(4)
gmvAlloc = pd.DataFrame(gmvArray, index=stockPort, columns=['Allocation']).T
gmvAlloc = gmvAlloc.T.sort_values(by=['Allocation'], ascending=True)

gmvAlloc.plot.bar()

plt.figure()
plt.scatter(results[0,:], results[1,:], c=results[2,:])
plt.colorbar()
plt.scatter(stdSharpe, returnSharpe, marker='*', color='r',s=500, label='MSR')
plt.scatter(stdGmv, returnGmv, marker='*', color='g', s=500, label='GMV')
plt.legend()
plt.title('Efficient Fronter - Markowitz Portfolios')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()
