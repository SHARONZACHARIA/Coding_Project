import pandas as pd 
import matplotlib as plt 
import numpy as np 

def readCsv():
    df = pd.read_csv('Dataset/data7.csv')
    return df

print(readCsv())