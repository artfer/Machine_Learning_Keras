import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from time import time
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


votesFeatures = ["Class","handicapped-infants","water-project-cost-sharing",
"adoption-of-the-budget-resolution","physician-fee-freeze",
"el-salvador-aid","religious-groups-in-schools",
"anti-satellite-test-ban","aid-to-nicaraguan-contras",
"mx-missile","immigration","synfuels-corporation-cutback",
"education-spending","superfund-right-to-sue","crime",
"duty-free-exports","export-administration-act-south-africa"]

def import_data(file):
    if 'votes' in file:
        df = pd.read_csv(file,names=votesFeatures)

        #turn into numeric values
        df[df.columns[1:]] = df[df.columns[1:]].applymap(lambda x: '1' if x=='y' else('0' if x=='n' else '-1'))
        df[df.columns[:1]] = df[df.columns[:1]].applymap(lambda x: '1' if x=='republican' else '0')

        for col in df[df.columns]:
            df[col] = pd.to_numeric(df[col])
    elif 'soccer' in file:
        df = pd.read_csv(file)

        header = df.iloc[0]
        df = df[1:]
        df.rename(columns = header)
        df = df.dropna()
        df['B365H'] = df['B365H'].map(lambda x : round(x,0))
        df['B365D'] = df['B365D'].map(lambda x : round(x,0))
        df['B365A'] = df['B365A'].map(lambda x : round(x,0))
    elif 'heart' in file:
        df = pd.read_csv(file)

        header = df.iloc[0]
        df = df[1:]
        df.rename(columns = header)
        df = df.rename(columns={'target': 'Class'})
    df = df.astype('float32')
    return df


def split_data(df):
    train,test = train_test_split(df,test_size=0.3) # 70% training and 30% test
    train_c = train['Class']
    train = train.drop('Class',axis=1)
    test_c = test['Class']
    test = test.drop('Class',axis=1)
    return train.values,train_c.values,test.values,test_c.values
