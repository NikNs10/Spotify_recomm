import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

data=pd.read_csv("C:\Users\Ssharma\Documents\Project\Spotify_recomm\dataset.csv")