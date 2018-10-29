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
    #a0,a1,x_mean,y_mean=calculate_a(data,0)
    #error_var,err,y_cap=calculate_error_var(data,0,a0,a1)
    data.columns=["X1","X2","X3","X4","X5","Y"]
    results = sm.OLS(data["Y"], sm.add_constant(data[["X1"]])).fit()
    print(results.summary())
    const=results.params['const']
    a1=results.params['X1']

    err=results.resid
    print("variance", np.var(err))

    slope, intercept, r_value, p_value, std_err = stats.linregress(data["X1"], data["Y"])
    print("\n slope: ",slope)
    print("\n intercept: ", intercept)
    print("\n p value: ", p_value)
    print("\n std_err: ", std_err)
    print("\n r-squared: ", r_value ** 2)

    plt.figure()
    line=slope*data["X1"]+intercept
    plt.plot(data["X1"],data["Y"],'o',data["X1"],line)
    plt.title('Regression Line')
    #plt.show()
    plt.savefig('Regression Line')

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

    y_cap=const+a1*data['X1']
    #print("y_cap",y_cap)

    plt.figure()
    plt.scatter(y_cap, err)
    plt.title("Scatter plot of residuals")
    plt.savefig("Scatter plot of residuals ")

    data.insert(loc=1,value=np.square(data["X1"]),column="X2n")


    print(data.head(3))

    results = sm.OLS(data["Y"], sm.add_constant(data[["X1","X2n"]])).fit()
    print(results.summary())
    const = results.params['const']
    a1 = results.params['X1']
    a2=results.params["X2n"]

    err = results.resid
    print("variance", np.var(err))

    fig = sm.qqplot(results.resid, line='q')
    plt.title("Q_Q plot of residuals against normal distribution ")
    # plt.show()
    plt.savefig("Q_Q plot")

    plt.figure()
    plt.hist(err, bins=10)
    plt.title("Residual Histogram ")
    plt.savefig("ResidualHist")

    chi_test = stats.normaltest(err)
    print("Z value", chi_test[0])
    print(" chi squared probability for the hypothesis test", chi_test[1])

    y_cap = const + a1 * data['X1']
    # print("y_cap",y_cap)

    plt.figure()
    plt.scatter(y_cap, err)
    plt.title("Scatter plot of residuals")
    plt.savefig("Scatter plot of residuals ")



