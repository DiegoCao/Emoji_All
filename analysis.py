import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df1 = pd.read_csv("avgemoji1.csv")
df2 = pd.read_csv("avgemoji2.csv")

df = df1.merge(df2, on='rid')
print(df.head())
x1 = df['avgemoji']
x2 = df['aidsum']
# plt.scatter(x1, x2, alpha=0.6

sns.regplot(x=x1,y=x2,x_estimator=np.mean, color='g', x_bins=20, fit_reg=False)

plt.xlabel('avg emoji (calculated by way 1)')
plt.ylabel('avg emoji (calculated by way 2) ')
plt.savefig('week4_4.png')
plt.show()

