import numpy as np
from scipy import stats
import argparse

parser = argparse.ArgumentParser(description='One-sample t-test')
parser.add_argument('subject')
parser.add_argument('-c', '--column', type=str, help='Column name for variable to test')
parser.add_argument('-v', '--value', type=str, help='Null hypothesis mean value')
known = parser.parse_args()

# Sample data (replace with your data)
data = f.load(known.subject)[known.value]

# Define the null hypothesis population mean (replace with your expected population mean)
null_mean = known.value

# Perform one-sample T-test
t_statistic, p_value = stats.ttest_1samp(data, null_mean)

# Print the results
print("T-statistic:", t_statistic)
print("P-value:", p_value)

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is sufficient evidence to conclude that the population mean is different from", null_mean)
else:
    print("Fail to reject the null hypothesis: There is not enough evidence to conclude that the population mean is different from", null_mean)

