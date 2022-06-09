# Explore file for Zillow project

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from scipy import stats 
import env
import wrangle
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler





#########################----------------------######################


# This function takes in a categorical variable and plots it against a continuous variable using a violin plot



def fips_vs_continuous_vars(df, continuous, categorical):

    plot_list = []
    for cat in categorical:
        for cont in continuous:
            plot_list.append([cat, cont])
    
    for i in plot_list:
        plt.figure(figsize=(18, 5))
        plt.subplot(131)
        sns.violinplot(x=i[0], y=i[1], data=df)
        plt.show()




