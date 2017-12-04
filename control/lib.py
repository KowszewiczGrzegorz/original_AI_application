from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression, RANSACRegressor, ElasticNet
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
# from keras.optimizers import Adam
# from keras.models import Sequential
# from keras.layers.core import Dense, Activation, Dropout, ActivityRegularization
# from keras.optimizers import SGD, RMSprop, Adadelta
# from keras.callbacks import EarlyStopping
# from keras.layers.advanced_activations import PReLU, LeakyReLU
# from keras.layers.normalization import BatchNormalization
# from keras.initializers import random_uniform
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def standardize_datas(X):
    """変数の標準化"""

    """もらったデータの標準化(平均=0, 標準偏差=1)を行う"""
    sc = StandardScaler()
    return_X = pd.DataFrame(sc.fit_transform(X))

    return return_X