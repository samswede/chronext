import pandas as pd
import matplotlib.pyplot as plt

def clean_data(file_path, date_col, data_col, new_data_col_name):
    """Loads data from a CSV file, parses dates, renames a specified column, and sets the date column as an index."""
    data = pd.read_csv(file_path)
    
    # Provide a specific date format if known, or improve error handling
    try:
        data[date_col] = pd.to_datetime(data[date_col], format='%a %b %d %Y')  # Modify format as per your data
    except ValueError:
        try:
            # Fallback for timezone or non-standard formats
            data[date_col] = pd.to_datetime(data[date_col].str[:-6])
        except ValueError as e:
            # Handle or log the error if datetime conversion fails entirely
            print(f"Error converting date: {e}")
            return None  # Or handle differently as needed
    
    data.rename(columns={data_col: new_data_col_name}, inplace=True)
    data.set_index(date_col, inplace=True)
    return data

def apply_percent_change_from_start(data):
    """Applies a percent change calculation to a DataFrame."""
    return data.div(data.iloc[0]).sub(1).mul(100)

def apply_exponential_smoothing(data, data_col, span):
    """Applies exponential smoothing to a specified column of data."""
    #print(data.head())
    return data[data_col].ewm(span=span, adjust=False).mean()

def apply_exponential_smoothing_v2(data, data_col_index: int, span):
    """Applies exponential smoothing to a specified column of data."""
    #print(data.head())  # Print the first 5 rows of the DataFrame
    return data.iloc[:, data_col_index].ewm(span=span, adjust=False).mean()

def plot_data(original_data, smoothed_data, labels, colors, title, xlabel, ylabel):
    """Plots original and smoothed data on a single y-axis with customizable colors."""
    plt.figure(figsize=(14, 8))

    for original in original_data:
        plt.plot(original.index, original, label=None, color=colors[0], alpha=0.3)

    for data, label, color in zip(smoothed_data, labels, colors[1:]):
        plt.plot(data.index, data, label=label, color=color)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_data_with_initial_and_final_values(original_data, smoothed_data, labels, colors, title, xlabel, ylabel):
    """Plots original and smoothed data on a single y-axis with customizable colors, emphasizing the first and final values."""
    plt.figure(figsize=(14, 8))

    # Plot original data with lower emphasis
    for original in original_data:
        plt.plot(original.index, original, label=None, color=colors[0], alpha=0.3)
        # Emphasize the first and last point
        plt.scatter([original.index[0], original.index[-1]], [original.iloc[0], original.iloc[-1]], color='green', zorder=5)

    # Plot smoothed data with labels
    for data, label, color in zip(smoothed_data, labels, colors[1:]):
        plt.plot(data.index, data, label=label, color=color)
        # Emphasize the first and last point
        plt.scatter([data.index[0], data.index[-1]], [data.iloc[0], data.iloc[-1]], color='green', zorder=5)
        # Add horizontal lines for the first and last values
        plt.hlines(y=[data.iloc[0], data.iloc[-1]], xmin=data.index[0], xmax=data.index[-1], color='green', linestyles='dashed', alpha=0.3)

        # Optionally add annotations
        plt.annotate(f'Start: {data.iloc[0]:.0f}', (data.index[0], data.iloc[0]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'End: {data.iloc[-1]:.0f}', (data.index[-1], data.iloc[-1]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_dual_axis(
        data_left, labels_left, 
        data_right, labels_right, 
        original_left, original_right, 
        title):
    """Plots multiple data series on a dual y-axis plot with specified color themes."""
    # Define color palettes
    warm_colors = ['#D55E00', '#E69F00', '#F0E442']  # Example warm colors: orange, gold, light yellow
    cool_colors = ['#56B4E9', '#0072B2', '#009E73']  # Example cool colors: sky blue, deep blue, teal

    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plotting the original unsmoothed data on the left axis in grey
    for data in original_left:
        ax1.plot(data.index, data, color='grey', alpha=0.2)
    
    # Plotting smoothed data on the left axis
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Watch Prices (CHF)', color='darkred')
    for data, label, color in zip(data_left, labels_left, warm_colors[:len(data_left)]):
        ax1.plot(data.index, data, label=label, color=color)
    ax1.tick_params(axis='y', labelcolor='darkred')
    ax1.legend(loc='upper left')

    # Setup the right axis
    ax2 = ax1.twinx()
    
    # Plotting the original unsmoothed data on the right axis in grey
    for data in original_right:
        ax2.plot(data.index, data, color='grey', alpha=0.2)
    
    ax2.set_ylabel('Market Indices (Index)', color='darkblue')
    for data, label, color in zip(data_right, labels_right, cool_colors[:len(data_right)]):
        ax2.plot(data.index, data, label=label, color=color)
    ax2.tick_params(axis='y', labelcolor='darkblue')
    ax2.legend(loc='upper right')

    plt.title(title)
    plt.grid(True)
    plt.show()

def main():
    file_path = 'SP500.csv'
    data_col = 'Close'
    date_col = 'Date'
    new_data_col_name = 'Close'
    
    # Load and clean data
    data = clean_data(file_path, date_col, data_col, new_data_col_name)
    
    # Apply smoothing
    spans = [30, 90, 365]
    smoothed_data = [apply_exponential_smoothing(data, data_col, span) for span in spans]
    labels = [f'EWMA {span} days' for span in spans]
    colors = ['grey', 'blue', 'purple', 'green']  # Example colors for original and each span
    
    # Plot data
    plot_data(data[data_col], smoothed_data, labels, colors, 'Smoothed S&P 500 with Various Spans', 'Date', 'Price')


def main_single_axis():
    sp500_data = clean_data('SP500.csv', 'Date', 'Close', 'Close')
    sp500_smoothed = apply_exponential_smoothing(sp500_data, 'Close', 90)
    plot_data(sp500_data['Close'], [sp500_smoothed], ['90-Day EWMA'], ['grey', 'purple'], 'SP500 90-Day EWMA', 'Date', 'Price')

def view_watch_data():
    audemars_data = clean_data('AudemarsPiguet.csv', 'date', 'Audemars Piguet 15500ST (CHF)', 'Close')
    rolex_data = clean_data('Rolex.csv', 'date', 'Rolex 116500 (CHF)', 'Close')
    patek_data = clean_data('PatekPhilippe.csv', 'date', 'Patek Philippe 5711/1A (CHF)', 'Close')

    print(audemars_data.head())
    print(rolex_data.head())
    print(patek_data.head())

    """
    Given that this is the structure:
    "
    date        Close        
    2019-05-13      0
    2019-05-14      0
    2019-05-15      0
    2019-05-16      0
    2019-05-17      0
    "

    Lets take only data after 2020-05-13
    
    """

    audemars_data = audemars_data[audemars_data.index >= '2020-05-13']
    print('Verify Audemars data is correct after filtering')
    print(audemars_data.head())

def date_of_first_value(df):
    """
    Function to find the date of the first non-zero value in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): Input DataFrame with a 'date' index and a 'Close' column containing numerical values.

    Returns:
    - str: Date of the first non-zero value in the 'Close' column (formatted as 'YYYY-MM-DD').
    """
    # Find the index of the first non-zero value in the 'Close' column
    first_non_zero_index = df[df['Close'] != 0].index[0]
    
    # Convert the index to string format and return
    return str(first_non_zero_index)

def process_market_index_data(file_names, smoothing_span, data_directory="./data/market"):
    results = {}
    
    for file_name in file_names:
        # Construct the file path
        file_path = f"{data_directory}/{file_name}.csv"
        
        # Construct variable names
        data_key = f"{file_name.lower()}_data"
        percent_change_key = f"{file_name.lower()}_percent_change"
        smoothed_key = f"{file_name.lower()}_smoothed"
        
        # Load Market Index Data
        data = clean_data(file_path, 'Date', 'Close', 'Close')
        percent_change = apply_percent_change_from_start(data)
        smoothed = apply_exponential_smoothing(percent_change, 'Close', smoothing_span)
        
        # Store results in the dictionary
        results[data_key] = data
        results[percent_change_key] = percent_change
        results[smoothed_key] = smoothed
    
    return results

def process_watch_data(file_names, smoothing_span, start_date, data_directory="./data/watches"):
    results = {}
    
    for file_name in file_names:
        # Construct the file path
        file_path = f"{data_directory}/{file_name}.csv"
        
        # Construct variable names
        data_key = f"{file_name.lower()}_data"
        filtered_data_key = f"{file_name.lower()}_filtered"
        percent_change_key = f"{file_name.lower()}_percent_change"
        smoothed_key = f"{file_name.lower()}_smoothed"
        
        # Load Watch Data
        data = clean_data(file_path, 'date', f'{file_name.replace("_", " ")} (CHF)', 'Close')
        
        # Identify the first non-zero value date
        start_date = date_of_first_value(data)

        # Filter data from start date
        filtered_data = data[data.index >= start_date]
        
        # Apply percent change
        percent_change = apply_percent_change_from_start(filtered_data)
        
        # Apply exponential smoothing
        smoothed = apply_exponential_smoothing(percent_change, 'Close', smoothing_span)
        
        # Store results in the dictionary
        results[data_key] = data
        results[filtered_data_key] = filtered_data
        results[percent_change_key] = percent_change
        results[smoothed_key] = smoothed
    
    return results


def main_percent_change():
    smoothing_span = 30

    cfr_data = clean_data('CFR.csv', 'Date', 'Close', 'Close')
    cfr_percent_change = apply_percent_change_from_start(cfr_data)
    cfr_smoothed = apply_exponential_smoothing(cfr_percent_change, 'Close', smoothing_span)

    uhr_data = clean_data('UHR.csv', 'Date', 'Close', 'Close')
    uhr_percent_change = apply_percent_change_from_start(uhr_data)
    uhr_smoothed = apply_exponential_smoothing(uhr_percent_change, 'Close', smoothing_span)

    # Load watches data
    audemars_data = clean_data('AudemarsPiguet.csv', 'date', 'Audemars Piguet 15500ST (CHF)', 'Close')
    rolex_data = clean_data('Rolex.csv', 'date', 'Rolex 116500 (CHF)', 'Close')
    patek_data = clean_data('PatekPhilippe.csv', 'date', 'Patek Philippe 5711/1A (CHF)', 'Close')

    audemars_4_years = audemars_data[audemars_data.index >= '2020-05-13']
    rolex_4_years = rolex_data[rolex_data.index >= '2020-05-13']
    patek_4_years = patek_data[patek_data.index >= '2020-05-13']

    # Apply percent change
    audemars_percent_change = apply_percent_change_from_start(audemars_4_years)
    rolex_percent_change = apply_percent_change_from_start(rolex_4_years)
    patek_percent_change = apply_percent_change_from_start(patek_4_years)

    # Apply exponential smoothing
    audemars_smoothed = apply_exponential_smoothing(audemars_percent_change, 'Close', smoothing_span)
    rolex_smoothed = apply_exponential_smoothing(rolex_percent_change, 'Close', smoothing_span)
    patek_smoothed = apply_exponential_smoothing(patek_percent_change, 'Close', smoothing_span)

        # Define color palettes
    warm_colors = ['#D55E00', '#E69F00', '#F0E442']  # Example warm colors: orange, gold, light yellow
    cool_colors = ['#56B4E9', '#0072B2', '#009E73']  # Example cool colors: sky blue, deep blue, teal

    labels = ['UHR', 'Audemars Piguet 15500ST', 'Rolex 116500', 'Patek Philippe 5711/1A']

    colors = ['grey', cool_colors[1], warm_colors[0], warm_colors[1], warm_colors[2] ]

    original_data = [ 
                        #cfr_percent_change['Close'],
                        uhr_percent_change['Close'],

                        audemars_percent_change,
                        rolex_percent_change,
                        patek_percent_change
                    ]
    
    smoothed_data = [
                        #cfr_smoothed,
                        uhr_smoothed,

                        audemars_smoothed,
                        rolex_smoothed,
                        patek_smoothed
                    ]
    
    title = f'{smoothing_span}-Day EWMA: Watches vs. Swatch Group (UHR) Percent Change'

    plot_data(original_data, smoothed_data, labels, colors, title, 'Date', '% Price Change')

def main_percent_change_v2():
    smoothing_span = 30

    # Process market index data
    market_indices = ['MXX']

    # Repurposing other function because this is a special case of non yahoo finance data
    #market_data = process_watch_data(market_indices, smoothing_span, '2020-05-13', data_directory="./data/market")
    # Otherwise, we would use the function below
    market_data = process_market_index_data(market_indices, smoothing_span)

    # Process watch data
    watch_models = [
                    'Breitling_RB0127', 
                    'Omega_310', 
                    'Cartier_WSSA0032', 
                    'IWC_324101', 
                    'Omega_310',
                    'Rolex_1675'
                    ]
    
    watch_data = process_watch_data(watch_models, smoothing_span, '2020-12-23')

    # Define color palettes
    warm_colors = ['#D55E00', '#E69F00', '#F0E442', '#009E73', '#56B4E9', '#0072B2']
    cool_colors = ['#56B4E9', '#0072B2', '#009E73']  # Example cool colors: sky blue, deep blue, teal

    labels = []
    #labels.extend(market_indices)
    labels.extend(watch_models)

    colors = ['grey']
    #colors.extend(cool_colors[:len(market_indices)])
    colors.extend(warm_colors[:len(watch_models)])
    

    original_data = [ 
                       # Market data
                        #market_data['mxx_percent_change'],

                        # Watches data
                        watch_data['breitling_rb0127_percent_change'],
                        watch_data['omega_310_percent_change'],
                        watch_data['cartier_wssa0032_percent_change'],
                        watch_data['iwc_324101_percent_change'],
                        watch_data['omega_310_percent_change'],
                        watch_data['rolex_1675_percent_change']
                    ]
    
    smoothed_data = [
                        # Market data
                        #market_data['mxx_smoothed'],

                        # Watches data
                        watch_data['breitling_rb0127_smoothed'],
                        watch_data['omega_310_smoothed'],
                        watch_data['cartier_wssa0032_smoothed'],
                        watch_data['iwc_324101_smoothed'],
                        watch_data['omega_310_smoothed'],
                        watch_data['rolex_1675_smoothed']
                    ]
    
    title = f'{smoothing_span}-Day EWMA: Watches Performance in Percent Change'

    plot_data(original_data, smoothed_data, labels, colors, title, 'Date', '% Price Change')




def main_percent_change_v3():
    smoothing_span = 30

    mxx_data = clean_data('./data/market/MXX.csv', 'Date', 'Close', 'Close')
    mxx_3_years = mxx_data[mxx_data.index >= '2022-12-23']
    mxx_percent_change = apply_percent_change_from_start(mxx_3_years)
    mxx_smoothed = apply_exponential_smoothing(mxx_percent_change, 'Close', smoothing_span)

    # Load watches data
    breitling_data = clean_data('./data/watches/Breitling_RB0127.csv', 'date', 'Breitling RB0127 (CHF)', 'Close')
    cartier_data = clean_data('./data/watches/Cartier_WSSA0032.csv', 'date', 'Cartier WSSA0032 (CHF)', 'Close')

    breitling_3_years = breitling_data[breitling_data.index >= '2022-12-23']
    cartier_3_years = cartier_data[cartier_data.index >= '2022-12-23']

    # Apply percent change
    breitling_percent_change = apply_percent_change_from_start(breitling_3_years)
    cartier_percent_change = apply_percent_change_from_start(cartier_3_years)

    # Apply exponential smoothing
    breitling_smoothed = apply_exponential_smoothing(breitling_percent_change, 'Close', smoothing_span)
    cartier_smoothed = apply_exponential_smoothing(cartier_percent_change, 'Close', smoothing_span)

    # Define color palettes

    # Six warm that between purple and yellow
    warm_colors = ['#D55E00', '#E69F00', '#F0E442', '#009E73', '#56B4E9', '#0072B2']    
    warm_colors = ['#D55E00', '#E69F00', '#F0E442', '#009E73', '#56B4E9', '#0072B2']
    cool_colors = ['#56B4E9', '#0072B2', '#009E73']  # Example cool colors: sky blue, deep blue, teal

    labels = ['Monaco', 'Breitling RB0127', 'Cartier WSSA0032']

    colors = ['grey', cool_colors[1], warm_colors[0], warm_colors[1], warm_colors[2], warm_colors[3], warm_colors[4], warm_colors[5] ]

    original_data = [ 
                        mxx_percent_change['Close'],

                        breitling_percent_change,
                        cartier_percent_change,
                    ]
    
    smoothed_data = [

                        mxx_smoothed,

                        breitling_smoothed,
                        cartier_smoothed,
                    ]
    
    title = f'{smoothing_span}-Day EWMA: Watches Performance in Percent Change'

    plot_data(original_data, smoothed_data, labels, colors, title, 'Date', '% Price Change')


def main_absolute_watches():
    smoothing_span = 30


    # Load watches data
    audemars_data = clean_data('AudemarsPiguet.csv', 'date', 'Audemars Piguet 15500ST (CHF)', 'Close')
    rolex_data = clean_data('Rolex.csv', 'date', 'Rolex 116500 (CHF)', 'Close')
    patek_data = clean_data('PatekPhilippe.csv', 'date', 'Patek Philippe 5711/1A (CHF)', 'Close')

    audemars_4_years = audemars_data[audemars_data.index >= '2020-05-13']
    rolex_4_years = rolex_data[rolex_data.index >= '2020-05-13']
    patek_4_years = patek_data[patek_data.index >= '2020-05-13']

    # Apply exponential smoothing
    audemars_smoothed = apply_exponential_smoothing(audemars_4_years, 'Close', smoothing_span)
    rolex_smoothed = apply_exponential_smoothing(rolex_4_years, 'Close', smoothing_span)
    patek_smoothed = apply_exponential_smoothing(patek_4_years, 'Close', smoothing_span)

        # Define color palettes
    warm_colors = ['#D55E00', '#E69F00', '#F0E442']  # Example warm colors: orange, gold, light yellow
    cool_colors = ['#56B4E9', '#0072B2', '#009E73']  # Example cool colors: sky blue, deep blue, teal

    labels = ['Audemars Piguet 15500ST', 'Rolex 116500', 'Patek Philippe 5711/1A']

    colors = ['grey', warm_colors[0], warm_colors[1], warm_colors[2] ]

    original_data = [ 
                        audemars_4_years,
                        rolex_4_years,
                        patek_4_years
                    ]
    
    smoothed_data = [
                        audemars_smoothed,
                        rolex_smoothed,
                        patek_smoothed
                    ]
    
    title = f'{smoothing_span}-Day EWMA: Watch Prices (CHF)'

    plot_data_with_initial_and_final_values(original_data, smoothed_data, labels, colors, title, 'Date', 'Price (CHF)')


def main_dual_axis():

    # Market data
    sp500_data = clean_data('SP500.csv', 'Date', 'Close', 'Close')
    ssmi_data = clean_data('SSMI.csv', 'Date', 'Close', 'Close')
    lvmh_data = clean_data('LVMH.csv', 'Date', 'Close', 'Close')

    # Watches data
    audemars_data = clean_data('AudemarsPiguet.csv', 'date', 'Audemars Piguet 15500ST (CHF)', 'Close')
    rolex_data = clean_data('Rolex.csv', 'date', 'Rolex 116500 (CHF)', 'Close')
    patek_data = clean_data('PatekPhilippe.csv', 'date', 'Patek Philippe 5711/1A (CHF)', 'Close')

    # Apply exponential smoothing
    smoothing_span = 30

    sp500_smoothed = apply_exponential_smoothing(sp500_data, 'Close', smoothing_span)
    ssmi_smoothed = apply_exponential_smoothing(ssmi_data, 'Close', smoothing_span)
    lvmh_smoothed = apply_exponential_smoothing(lvmh_data, 'Close', smoothing_span)

    audemars_smoothed = apply_exponential_smoothing(audemars_data, 'Close', smoothing_span)
    rolex_smoothed = apply_exponential_smoothing(rolex_data, 'Close', smoothing_span)
    patek_smoothed = apply_exponential_smoothing(patek_data, 'Close', smoothing_span)

    # Generate labels dynamically
    market_labels = ['SSMI']
    watch_labels = ['Audemars Piguet 15500ST', 'Rolex 116500', 'Patek Philippe 5711/1A']
    market_smoothed = [ssmi_smoothed]
    watch_smoothed = [audemars_smoothed, rolex_smoothed, patek_smoothed]
    original_market = [ssmi_data['Close']]
    original_watches = [audemars_data['Close'], rolex_data['Close'], patek_data['Close']]

    # Call the plotting function with additional original data series
    plot_dual_axis(
        watch_smoothed,
        [f'{watch} ({smoothing_span}-Day EWMA)' for watch in watch_labels],
        market_smoothed,
        [f'{market} ({smoothing_span}-Day EWMA)' for market in market_labels],
        original_watches,
        original_market,
        f'{smoothing_span}-Day EWMA: Watches vs. Market Indices'
    )


if __name__ == '__main__':
    main_percent_change_v2()
