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

data_sug = data_result.sort_values(by=['euclidean']).iloc[:6]
data_big = data_sort.set_index(data_sort.loc[:, 'track_id'])
track_list = pd.DataFrame()
for i in list(data_sug.loc[:, 'track_id']):
    if i in list(data_sort.loc[:, 'track_id']):
        track_info = data_big.loc[[i], ['track_name', 'artists']]
        #track_list = track_list.append(track_info)
        track_list = pd.concat([track_list, track_info], ignore_index=True)

recomended = track_list.values.tolist()
print(f"""You've just listened:   {recomended[0][0]} - {recomended[0][1]}
Now you may listen :
'{recomended[1][0]} - {recomended[1][1]}'
Or any of:
'{recomended[2][0]} - {recomended[2][1]}'
'{recomended[3][0]} - {recomended[3][1]}'
'{recomended[4][0]} - {recomended[4][1]}'
'{recomended[5][0]} - {recomended[5][1]}'  """)




