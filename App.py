import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

#CSV_Data Loaded
data=pd.read_csv("C:\Users\Ssharma\Documents\Project\Spotify_recomm\dataset.csv")

#To print first 5 rows of the data
print(data.head())


#Data Preprocessing

#dropping null  and duplicate values
df = data.drop("Unnamed: 0", axis=1)
df.isna().sum()
df = df.dropna()
df.isna().sum()
df = df.drop_duplicates()

df.info()
df.shape
popular_artists = df.groupby("artists").count().sort_values(by='popularity', ascending=False)['popularity'][:5]
popular_artists

#Data Visualization

corr_table = df.corr(method="pearson")
plt.figure(figsize=(16,4))
sns.heatmap(corr_table, annot=True, fmt=".1g")
plt.title("Correlation Heatmap between variables")
plt.show()



