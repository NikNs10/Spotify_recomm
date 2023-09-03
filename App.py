import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from scipy.spatial import distance

#CSV_Data Loaded
data=pd.read_csv("dataset.csv")

#To print first 5 rows of the data
print(data.head())




#Dropping Null  and Duplicate Values
df = data.drop("Unnamed: 0", axis=1)
df.isna().sum()
df = df.dropna()
df.isna().sum()
df = df.drop_duplicates()

#Data Visualization

corr_table = df.corr(method="pearson")
plt.figure(figsize=(16,4))
sns.heatmap(corr_table, annot=True, fmt=".1g")
plt.title("Correlation Heatmap between variables")
plt.show()

df.info()
df.shape
popular_artists = df.groupby("artists").count().sort_values(by='popularity', ascending=False)['popularity'][:5]
popular_artists

data_sort = data.drop(['time_signature', 'Unnamed: 0', 'key'], axis=1)
data_sort.drop_duplicates(subset=['track_id'], inplace=True)

#Data Preprocessing
scaler = preprocessing.MinMaxScaler()
names = data_sort.select_dtypes(include = np.number).columns
d = scaler.fit_transform(data_sort.select_dtypes(include = np.number))
data_norm = pd.DataFrame(d, columns=names)
data_norm.set_index(data_sort.loc[:, 'track_id'], inplace=True)

x = list(data_norm.iloc[19])


data_result = pd.DataFrame()
data_result['euclidean'] = [distance.euclidean(obj, x) for index, obj in data_norm.iterrows()]
data_result['track_id'] = data_norm.index






