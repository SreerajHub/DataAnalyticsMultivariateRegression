import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Read provided data as dataframe
data=pd.read_csv("srajend2.csv",header=-1)
#print(data.head(3))
#print(data[5][0])

for i in range(data.shape[1]-1):
    mean = np.mean(data[i])
    print("mean X"+str(i)+": "+ str(mean))

    var = np.var(data[i])
    print("Variance of X"+str(i)+": "+ str(var))

    plt.figure()
    plt.hist(data[i], bins='auto')
    plt.title("Histogram for X"+str(i))
    plt.show()
    
    plt.figure(2)
    plt.boxplot(data[i])
    plt.title("BoxPlot for X" + str(i))
    plt.show()
correlation=data.corr()
print("Correlation Matrix: \n", correlation)

#Remove outliers
data=data[(np.abs(stats.zscore(data)) < 2.5).all(axis=1)]
#Box plot after removing outliers
for i in range(0,data.shape[1]-1,1):
    plt.figure()
    plt.boxplot(data[i])
    plt.title("BoxPlot for X" + str(i))
    plt.show()

correlation=data.corr()
print("Correlation Matrix after removing outliers: \n", correlation)
