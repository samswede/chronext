import pandas as pd
import matplotlib.pyplot as plt


#TODO: Fix main functions iteratively to ensure they run without errors and produce the expected output.

#=========================================================================================================
# Modular Functions
#=========================================================================================================

def clean_data(file_path, date_col, data_col, new_data_col_name):
    """
    Loads and cleans data from a specified CSV file.

    Args:
        file_path (str): Path to the CSV file.
        date_col (str): The name of the column containing date information.
        data_col (str): The name of the column containing the data to be processed.
        new_data_col_name (str): New name for the data column.

    Returns:
        pd.DataFrame: A DataFrame with the date column set as the index and renamed data column.
    """
    data = pd.read_csv(file_path)
    
    try:
        # Attempt to parse dates in the specified format
        data[date_col] = pd.to_datetime(data[date_col], format='%a %b %d %Y')
    except ValueError:
        try:
            # Fallback for timezone or non-standard formats
            data[date_col] = pd.to_datetime(data[date_col].str[:-6])
        except ValueError as e:
            print(f"Error converting date: {e}")
            return None
    
    data.rename(columns={data_col: new_data_col_name}, inplace=True)
    data.set_index(date_col, inplace=True)
    return data

def apply_percent_change_from_start(data):
    """
    Calculates the percent change from the start for each column in the DataFrame.

    Args:
        data (pd.DataFrame): The data for which the percent change is to be calculated.

    Returns:
        pd.DataFrame: DataFrame with percent changes from the start.
    """
    return data.div(data.iloc[0]).sub(1).mul(100)

def apply_exponential_smoothing(data, data_col, span):
    """
    Applies exponential smoothing to a specified column of data.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        data_col (str): The column to apply smoothing to.
        span (int): The span for the exponential smoothing (controls the degree of smoothing).

    Returns:
        pd.Series: Smoothed data series.
    """
    return data[data_col].ewm(span=span, adjust=False).mean()

def plot_data(original_data, smoothed_data, labels, colors, title, xlabel, ylabel):
    """
    Plots original and smoothed data on a single y-axis with customizable colors and labels.

    Args:
        original_data (list of pd.Series): List of original data series.
        smoothed_data (list of pd.Series): List of smoothed data series.
        labels (list of str): Labels for each smoothed data series.
        colors (list of str): Colors for each data series.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(14, 8))

    # Plot original data with less emphasis
    for original in original_data:
        plt.plot(original.index, original, label=None, color=colors[0], alpha=0.3)

    # Plot smoothed data with labels
    for data, label, color in zip(smoothed_data, labels, colors[1:]):
        plt.plot(data.index, data, label=label, color=color)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_data_with_initial_and_final_values(original_data, smoothed_data, labels, colors, title, xlabel, ylabel):
    """
    Plots original and smoothed data on a single y-axis with customizable colors, emphasizing the first and final values.

    Args:
        original_data (list of pd.Series): List of original data series.
        smoothed_data (list of pd.Series): List of smoothed data series.
        labels (list of str): Labels for each smoothed data series.
        colors (list of str): Colors for each data series.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(14, 8))

    # Plot original data with less emphasis
    for original in original_data:
        plt.plot(original.index, original, label=None, color=colors[0], alpha=0.3)
        plt.scatter([original.index[0], original.index[-1]], [original.iloc[0], original.iloc[-1]], color='green', zorder=5)

    # Plot smoothed data with labels and emphasize the first and last points
    for data, label, color in zip(smoothed_data, labels, colors[1:]):
        plt.plot(data.index, data, label=label, color=color)
        plt.scatter([data.index[0], data.index[-1]], [data.iloc[0], data.iloc[-1]], color='green', zorder=5)
        plt.hlines(y=[data.iloc[0], data.iloc[-1]], xmin=data.index[0], xmax=data.index[-1], color='green', linestyles='dashed', alpha=0.3)
        plt.annotate(f'Start: {data.iloc[0]:.0f}', (data.index[0], data.iloc[0]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'End: {data.iloc[-1]:.0f}', (data.index[-1], data.iloc[-1]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_dual_axis(data_left, labels_left, data_right, labels_right, original_left, original_right, title):
    """
    Plots multiple data series on a dual y-axis plot with specified color themes.

    Args:
        data_left (list of pd.Series): Smoothed data series for the left y-axis.
        labels_left (list of str): Labels for the left y-axis data.
        data_right (list of pd.Series): Smoothed data series for the right y-axis.
        labels_right (list of str): Labels for the right y-axis data.
        original_left (list of pd.Series): Original unsmoothed data for the left y-axis.
        original_right (list of pd.Series): Original unsmoothed data for the right y-axis.
        title (str): Title of the plot.
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plotting the original unsmoothed data on the left axis in grey
    for data in original_left:
        ax1.plot(data.index, data, color='grey', alpha=0.2)

    # Plotting smoothed data on the left axis
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Left Axis Data', color='darkred')  # Update label description as per actual data context
    for data, label, color in zip(data_left, labels_left, ['#D55E00', '#E69F00', '#F0E442'][:len(data_left)]):
        ax1.plot(data.index, data, label=label, color=color)
    ax1.tick_params(axis='y', labelcolor='darkred')
    ax1.legend(loc='upper left')

    # Setup the right axis for additional data series
    ax2 = ax1.twinx()
    ax2.set_ylabel('Right Axis Data', color='darkblue')  # Update label description as per actual data context
    for data, label, color in zip(data_right, labels_right, ['#56B4E9', '#0072B2', '#009E73'][:len(data_right)]):
        ax2.plot(data.index, data, label=label, color=color)
    ax2.tick_params(axis='y', labelcolor='darkblue')
    ax2.legend(loc='upper right')

    plt.title(title)
    plt.grid(True)
    plt.show()

def view_watch_data():
    """
    Function to load, clean, and display initial rows of watch data for different brands, and filters data from a specific date onwards.
    """
    # Load and clean data for three different watch brands
    audemars_data = clean_data('AudemarsPiguet.csv', 'date', 'Audemars Piguet 15500ST (CHF)', 'Close')
    rolex_data = clean_data('Rolex.csv', 'date', 'Rolex 116500 (CHF)', 'Close')
    patek_data = clean_data('PatekPhilippe.csv', 'date', 'Patek Philippe 5711/1A (CHF)', 'Close')

    if not all([audemars_data, rolex_data, patek_data]):
        print("Error loading one or more datasets. Check file paths and formats.")
        return

    # Display head of data for verification
    print(audemars_data.head())
    print(rolex_data.head())
    print(patek_data.head())

    # Filter data from a specific date
    cutoff_date = '2020-05-13'
    audemars_data = audemars_data[audemars_data.index >= cutoff_date]
    print('Verify Audemars data is correct after filtering')
    print(audemars_data.head())


#=========================================================================================================
# Main Scripts
#=========================================================================================================

def main():
    """
    Main function to demonstrate cleaning data and plotting smoothed data for S&P 500 with different spans.
    """
    file_path = 'SP500.csv'
    data_col = 'Close'
    date_col = 'Date'
    new_data_col_name = 'Close'
    
    # Load and clean data
    data = clean_data(file_path, date_col, data_col, new_data_col_name)
    if data is None:
        print("Data loading failed. Check input files and formats.")
        return
    
    # Apply smoothing with different spans
    spans = [30, 90, 365]
    smoothed_data = [apply_exponential_smoothing(data, data_col, span) for span in spans]
    labels = [f'EWMA {span} days' for span in spans]
    colors = ['grey', 'blue', 'purple', 'green']  # Colors for original and each smoothed series
    
    # Plot the data
    plot_data(data[data_col], smoothed_data, labels, colors, 'Smoothed S&P 500 with Various Spans', 'Date', 'Price')

def main_single_axis():
    """
    Demonstrates the loading, cleaning, smoothing, and plotting of S&P 500 data for a single smoothing span.
    """
    sp500_data = clean_data('SP500.csv', 'Date', 'Close', 'Close')
    if sp500_data is None:
        print("Failed to load S&P 500 data.")
        return
    
    # Apply exponential smoothing for a 90-day span
    sp500_smoothed = apply_exponential_smoothing(sp500_data, 'Close', 90)
    
    # Plot the original and smoothed data
    plot_data(sp500_data['Close'], [sp500_smoothed], ['90-Day EWMA'], ['grey', 'purple'], 'SP500 90-Day EWMA', 'Date', 'Price')

def main_percent_change():
    """
    Demonstrates loading, cleaning, calculating percent change, smoothing, and plotting percent changes for watch data.
    """
    smoothing_span = 30
    # Load and clean data for multiple datasets
    watch_data_paths = {
        'AudemarsPiguet': 'AudemarsPiguet.csv',
        'Rolex': 'Rolex.csv',
        'PatekPhilippe': 'PatekPhilippe.csv'
    }
    watch_data = {}
    for brand, path in watch_data_paths.items():
        cleaned_data = clean_data(path, 'date', f'{brand} (CHF)', 'Close')
        if cleaned_data is None:
            print(f"Failed to load data for {brand}.")
            continue
        watch_data[brand] = cleaned_data[cleaned_data.index >= '2020-05-13']
    
    if not watch_data:
        print("No watch data was loaded successfully.")
        return

    # Apply percent change from start and exponential smoothing
    for brand, data in watch_data.items():
        percent_change = apply_percent_change_from_start(data)
        smoothed = apply_exponential_smoothing(percent_change, 'Close', smoothing_span)
        watch_data[brand] = (percent_change, smoothed)
    
    # Plotting
    original_data = [data[0] for data in watch_data.values()]
    smoothed_data = [data[1] for data in watch_data.values()]
    labels = list(watch_data.keys())
    colors = ['grey', 'orange', 'blue', 'purple']

    title = f'{smoothing_span}-Day EWMA: Watch Price Changes'
    plot_data(original_data, smoothed_data, labels, colors, title, 'Date', '% Price Change')

def main_absolute_watches():
    """
    Main function to demonstrate exponential smoothing applied to luxury watch price data and plotting with initial and final values emphasized.
    """
    # Define the span for smoothing
    smoothing_span = 30

    # Load and clean watch data for different brands
    audemars_data = clean_data('AudemarsPiguet.csv', 'date', 'Audemars Piguet 15500ST (CHF)', 'Close')
    rolex_data = clean_data('Rolex.csv', 'date', 'Rolex 116500 (CHF)', 'Close')
    patek_data = clean_data('PatekPhilippe.csv', 'date', 'Patek Philippe 5711/1A (CHF)', 'Close')

    if not all([audemars_data, rolex_data, patek_data]):
        print("Error loading one or more datasets.")
        return

    # Filter data to include only recent years
    cutoff_date = '2020-05-13'
    audemars_4_years = audemars_data[audemars_data.index >= cutoff_date]
    rolex_4_years = rolex_data[rolex_data.index >= cutoff_date]
    patek_4_years = patek_data[patek_data.index >= cutoff_date]

    # Apply exponential smoothing to the data
    audemars_smoothed = apply_exponential_smoothing(audemars_4_years, 'Close', smoothing_span)
    rolex_smoothed = apply_exponential_smoothing(rolex_4_years, 'Close', smoothing_span)
    patek_smoothed = apply_exponential_smoothing(patek_4_years, 'Close', smoothing_span)

    # Define labels and colors for plotting
    labels = ['Audemars Piguet 15500ST', 'Rolex 116500', 'Patek Philippe 5711/1A']
    colors = ['grey', '#D55E00', '#E69F00', '#F0E442']  # Example colors for original and smoothed series

    # Organize data for plotting
    original_data = [audemars_4_years['Close'], rolex_4_years['Close'], patek_4_years['Close']]
    smoothed_data = [audemars_smoothed, rolex_smoothed, patek_smoothed]

    # Plotting
    title = f'{smoothing_span}-Day EWMA: Luxury Watch Prices (CHF)'
    plot_data_with_initial_and_final_values(original_data, smoothed_data, labels, colors, title, 'Date', 'Price (CHF)')

def main_dual_axis():
    """
    Main function to demonstrate plotting smoothed data for market indices and watch prices on dual y-axes.
    """
    # Load and clean data for market indices and luxury watches
    market_data_paths = {
        'SP500': 'SP500.csv',
        'SSMI': 'SSMI.csv',
        'LVMH': 'LVMH.csv'
    }
    watch_data_paths = {
        'AudemarsPiguet': 'AudemarsPiguet.csv',
        'Rolex': 'Rolex.csv',
        'PatekPhilippe': 'PatekPhilippe.csv'
    }

    # Define smoothing span
    smoothing_span = 30

    # Clean and prepare data
    market_data, watch_data = {}, {}
    for name, path in {**market_data_paths, **watch_data_paths}.items():
        data = clean_data(path, 'Date' if name in market_data_paths else 'date', 'Close', 'Close')
        if data is None:
            print(f"Error loading data for {name}.")
            continue

        # Apply exponential smoothing
        smoothed = apply_exponential_smoothing(data, 'Close', smoothing_span)
        if name in market_data_paths:
            market_data[name] = (data['Close'], smoothed)
        else:
            watch_data[name] = (data['Close'], smoothed)

    # Prepare labels for plotting
    market_labels = [f'{name} ({smoothing_span}-Day EWMA)' for name in market_data]
    watch_labels = [f'{name} ({smoothing_span}-Day EWMA)' for name in watch_data]

    # Prepare data for plotting
    original_market = [data[0] for data in market_data.values()]
    market_smoothed = [data[1] for data in market_data.values()]
    original_watches = [data[0] for data in watch_data.values()]
    watch_smoothed = [data[1] for data in watch_data.values()]

    # Plot using dual axes
    plot_dual_axis(
        watch_smoothed,
        watch_labels,
        market_smoothed,
        market_labels,
        original_watches,
        original_market,
        f'{smoothing_span}-Day EWMA: Watches vs. Market Indices'
    )


if __name__ == '__main__':
    # This is where you can call the main function you want to run
    main_single_axis()
