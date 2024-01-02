import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import gaussian_kde

path = 'Dataset/data7.csv'
column_name = 'Annual_Salary'

#Function to read dataset
def readCsv(path,column_name):
    df = pd.read_csv(path, names=[column_name])
    df = df['Annual_Salary']
    return df

#Function for Probability Density Function
def CalculateMeanAndPDF(data):
    kde = gaussian_kde(data) 
    min_val = min(data)
    max_val = max(data)
    x_vals = np.linspace(min_val, max_val, 1000)
    pdf_values = kde.pdf(x_vals)
    # Calculate mean using the data
    mean = np.mean(data)
    return x_vals, pdf_values, mean

#Function to calculate fraction(X) of Population with salaries between 0.75W & W
def CalculateFraction(data ,mean):
    # Calculate the fraction of the population between 0.75W̃ and W̃
    lower_bound = 0.75 * mean
    upper_bound = mean
    
    # Calculate the fraction between 0.75W̃ and W̃ 
    fraction = upper_bound - lower_bound
    return fraction

#Function to Plot Histogram from PDF 
def PlotHist(data,mean,fraction):
    plt.figure(figsize=(8, 6))
    plt.hist(data,bins=30 ,density=True, alpha=0.7, color='skyblue', edgecolor='black',label='Probability Density Function (PDF)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Probability Density Function')
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean Salary ($W̃$): {mean:.2f}')
    plt.axvline(fraction, color='green', linestyle='dashed', linewidth=1, label=f'X: {fraction:.2f}')
    plt.text(0.5, 0.95, f'Mean Salary ($W̃$): {mean:.2f}', ha='center', va='center', transform=plt.gca().transAxes, color='red')
    plt.text(0.5, 0.90, f'X: {fraction:.2f}', ha='center', va='center', transform=plt.gca().transAxes, color='green')
    plt.legend()
    plt.show()
    

#Main Function 
def AnalysePDF():
    data = readCsv(path,column_name)
    X_vals,pdf_vals,mean = CalculateMeanAndPDF(data)
    fraction = CalculateFraction(data, mean)
    PlotHist(data,mean,fraction)


AnalysePDF()