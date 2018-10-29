import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chisquare
import statsmodels.api as sm
from pandas.tools import plotting


if __name__=="__main__":
    # Read provided data as dataframe
    data = pd.read_csv("srajend2.csv", header=-1)
    data = data[(np.abs(stats.zscore(data)) < 2.5).all(axis=1)]
    y = data.shape[1] - 1
    data.columns=["X1","X2","X3","X4","X5","Y"]
    results = sm.OLS(data["Y"], sm.add_constant(data[["X1","X4","X5"]])).fit()
    print(results.summary())
    const=results.params['const']
    a1=results.params['X1']
    #a2 = results.params['X2']
    #a3 = results.params['X3']
    a4 = results.params['X4']
    a5 = results.params['X5']


    #Task3.3 Q_Q plot
    err=results.resid
    fig=sm.qqplot(results.resid, line='q')
    plt.title("Q_Q plot of residuals against normal distribution ")
    #plt.show()
    plt.savefig("Q_Q plot")


    plt.figure()
    plt.hist(err, bins=10)
    plt.title("Residual Histogram ")
    plt.savefig("ResidualHist")

    chi_test = stats.normaltest(err)
    print("Z value", chi_test[0])
    print(" chi squared probability for the hypothesis test", chi_test[1])
    y_cap=const+a1*data['X1']+a4*data['X4']+a5*data['X5']
    #print("y_cap",y_cap)

    plt.figure()
    plt.scatter(y_cap, err)
    plt.title("Scatter plot of residuals")
    plt.savefig("Scatter plot of residuals ")
