import numpy as np
import scipy.stats as stats
mean_1 = 50
mean_2 = 45
std_1 = 10
std_2 = 12
size_1 = 40
size_2 = 35
z_score_two_sample = (mean_1 - mean_2) / np.sqrt((std_1**2 / size_1) + (std_2**2 / size_2))
p_value_two_sample = 2 * (1 - stats.norm.cdf(abs(z_score_two_sample)))
print(f"Z-Score: {z_score_two_sample}")
print(f"P-value: {p_value_two_sample}")
