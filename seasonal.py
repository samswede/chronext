import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

# Load the data
file_path = '/path_to_your_file.csv'  
data = pd.read_csv(file_path)

# Convert the 'date' column to datetime format for easier manipulation
data['date'] = pd.to_datetime(data['date'], format='%a %b %d %Y')

# Set the date column as the index
data.set_index('date', inplace=True)

# Resample the data to get monthly averages
monthly_data = data.resample('M').mean()  # Resample to monthly frequency and calculate mean

# Perform seasonal decomposition
decomposition = seasonal_decompose(monthly_data['Patek Philippe 5711/1A (CHF)'], model='additive')

# Plot the decomposed components
decomposed_fig = decomposition.plot()
decomposed_fig.set_size_inches(14, 10)
plt.show()

# Plot the Autocorrelation Function (ACF) for the monthly data
plt.figure(figsize=(14, 5))
plot_acf(monthly_data['Patek Philippe 5711/1A (CHF)'], lags=24, alpha=0.05)  # Analyzing up to 2 years of lags
plt.title('Autocorrelation Function (ACF) for Monthly Average Prices')
plt.xlabel('Lags in Months')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()
