import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde 

# Read data from CSV file
data = pd.read_csv('data7.csv',header=None,names=['Salary'])

# Calculate probability density function using Gaussian KDE
kde = gaussian_kde(data['Salary'], bw_method=0.5)  # Adjust bandwidth for KDE

# Generate values for plotting PDF
x_vals = np.linspace(data['Salary'].min(), data['Salary'].max(), 1000)
pdf = kde.evaluate(x_vals)

# Calculate mean annual salary
salaries = data['Salary']
mean_salary = np.mean(data['Salary'])
mean_salary = round(mean_salary, 2)
lower_bound = 0.75 * mean_salary
upper_bound = mean_salary


#function to calculate required value 'X' (fraction of population between 0.75W~ and W~)
def calculate_Fraction(salaries):
    """
    Function to calculate the fraction of population  between 0.75W~ and W~
    Args:
        salaries (dataframe): 

    Returns:
        float: decimal value with Value of X .
    """
    x_values = salaries[(salaries >= lower_bound) & (salaries <= upper_bound)]
    fraction_population = len(x_values) / len(salaries)
    fraction_population_rounded = round(fraction_population, 2)
    return fraction_population_rounded


# Function to get the statistical description of the dataset 
def getStatisticalDescription(data):
    mean = np.mean(data)
    median = np.median(data)
    mode = data.mode()
    std = data.std()
    kurtosis = data.kurtosis()
    skewness = data.skew()
    range = max(data) - min(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    iqr = Q3 - Q1
    return  mean,mode,median, std,kurtosis,skewness, range , max(data) , min(data) , iqr


mean,mode,median, std,kurtosis,skewness, range  , maxV , minV , iqr = getStatisticalDescription(data['Salary'])
print(f"mean : {mean}, mode is : {mode}, median : {median}, std : {std}, kurtosis : {kurtosis}, skewness : {skewness},Range : {range} , Max : {maxV} , Min : {minV} , IQR {iqr}")


# Plotting histogram and PDF
def plotHist(data,mean_salary,pdf):

    """
    Generates a Histogram with PDF , mean value and value of X 
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data['Salary'], bins=30, density=True, alpha=0.7, label='Histogram',edgecolor='black')
    plt.plot(x_vals, pdf, label='Probability Density Function',)
    plt.axvline(mean_salary, color='red', linestyle='dashed', linewidth=2, label=f'Mean Salary ($\~{{W}}$): {mean_salary}')
    plt.axvspan(lower_bound, upper_bound, color='green', alpha=0.3, label=f'Population Fraction (X): {calculate_Fraction(salaries)}')
    plt.xlabel('Salary (Euros)')
    plt.ylabel('Density')
    plt.title('Probability Density Function of Salaries')
    plt.xlim(0,plt.xlim()[1])
    plt.ylim(0,plt.ylim()[1])
    plt.legend()
    plt.tight_layout()
    plt.show()


plotHist(data,mean_salary,pdf)

