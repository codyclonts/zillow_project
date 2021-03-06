import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from env import get_db_url
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler





##### acquire zillow data ######

def get_zillow_data():
    filename = 'zillow_data.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else:
        df = pd.read_sql(
            '''
            SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips, parcelid  FROM properties_2017 
            JOIN propertylandusetype USING (propertylandusetypeid) 
            JOIN predictions_2017 USING (parcelid)
            WHERE propertylandusedesc = 'Single Family Residential'; 
            '''
            ,
            get_db_url('zillow')
        )

        df.to_csv(filename)

        return df


def handle_nulls(df):
    df = df.dropna()
    return df



def make_int(df):
    df = df.astype({'bedroomcnt':'int', 'calculatedfinishedsquarefeet':'int', 'taxvaluedollarcnt':'int', 'yearbuilt':'int','fips':'int'})
    return df




def handle_outliers(df):
    df = df[df.bathroomcnt <= 6]

    df = df[df.bathroomcnt >= 1]
    
    df = df[df.bedroomcnt <= 6]

    df = df[df.bedroomcnt >= 1]

    df = df[df.taxvaluedollarcnt < 4_000_000]

    df = df[df.calculatedfinishedsquarefeet < 100000]

    return df


def split_zillow_data(df):

    train_validate, test = train_test_split(df, test_size=.2, 
        random_state=123)

    train, validate = train_test_split(train_validate, test_size=.3, 
        random_state=123)
    return train, validate, test


def fips2(fips):
  if fips == 'LA':
    return 6037
  elif fips == 'Orange':
    return 6059
  else:
    return 6111

def wrangle_zillow():
    df= get_zillow_data()
    df = handle_nulls(df)
    df = make_int(df)
    df = handle_outliers(df)
    df['fips'].replace({6037 : 'LA' , 6059 : 'Orange' , 6111: 'Ventura'}, inplace = True)
    df['age'] = 2022 - df.yearbuilt
    df["fips2"] = df["fips"].apply(lambda fips: fips2(fips))
    return df 


def data_scaled(train, validate, test, columns_to_scale):
    '''
    This function takes in train, validate, test subsets of the cleaned zillow dataset and using the train subset creates a min_max 
    scaler. It thens scales the subsets and returns the train, validate, test subsets as scaled versions of the initial data.
    Arguments:  train, validate, test - split subsets from of the cleaned zillow dataframe
                columns_to_scale - a list of column names to scale
    Return: scaled_train, scaled_validate, scaled_test - dataframe with scaled versions of the initial unscaled dataframes 
    '''
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.fit_transform(train[columns_to_scale]), 
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])

    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])

    return train_scaled, validate_scaled, test_scaled


