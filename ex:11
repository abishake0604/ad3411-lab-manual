import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
date_range = pd.date_range(start='1/1/2020', periods=100)
data = np.random.randn(100).cumsum()
time_series_data = pd.DataFrame(data, index=date_range, columns=['Value'])
plt.figure(figsize=(12, 6))
plt.plot(time_series_data.index, time_series_data['Value'], label='Random Data', color='blue')
plt.title('Time Series Analysis')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()
