import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
def load_df():
    df_ = pd.read_csv('datasets/train.csv')
    return df_


df = load_df()
df.head()
