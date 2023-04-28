import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframes = []
files = ['appearances.csv', 'club_games.csv', 'clubs.csv', 'competitions.csv',
         'game_events.csv', 'games.csv', 'player_valuations.csv', 'players.csv']
def file_func():
    for f in files:
        df = pd.read_csv(f)
        print(f, df.shape)
        dataframes.append(df)
    print('Files processed')

def growth_in_value():
    df = dataframes[7]
    # Define a list of league codes
    league_codes = ['GB1', 'IT1', 'ES1', 'L1', 'FR1']

    # Create a dictionary of DataFrames for each league
    league_dfs = {}
    latest_dfs = {}
    for code in league_codes:
        league_dfs[code] = df[df['current_club_domestic_competition_id'] == code]
    for code, l_df in league_dfs.items():
        latest_dfs[code] = l_df[l_df['last_season'] == 2022]

    # Iterate over the dictionary and extract the information you need
    m_totals = {}
    for code, l_df in league_dfs.items():
        total_value = l_df['highest_market_value_in_eur'].sum(skipna=True)
        m_totals[code] = total_value / 1000000000
    l_totals = {}
    for code, latest_df in latest_dfs.items():
        total_value = latest_df['highest_market_value_in_eur'].sum(skipna=True)
        l_totals[code] = total_value / 1000000000

    # Define the league codes and their positions on the x-axis
    leagues = ['Premier League', 'Serie A', 'LaLiga', 'Bundesliga', 'Ligue 1']

    positions = np.arange(len(leagues))

    # Create arrays with the total values for each league
    m_values = [m_totals[code] for code in league_codes]
    l_values = [l_totals[code] for code in league_codes]

    # Plot the bars side by side
    width = 0.4

    plt.bar(positions + width / 2, l_values, width=width, color='r', label='Players from 2022')
    plt.bar(positions - width / 2, m_values, width=width, color='g', label='Players from 2012-2022')

    # Set the axis labels, title, and legend
    plt.xlabel('Top 5 European Leagues')
    plt.ylabel('Total Market Value (billions of euros)')
    plt.title('Total Market Value of Players in Top 5 leagues')
    plt.xticks(positions, leagues)
    plt.legend()

    # Show the plot
    plt.show()

         
def transfer_over_time():
    # load the dataset and create year column
    df = dataframes[6]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year

    # create list of quarters and dataframes for each quarter
    quarters = [(2003, 2007), (2008, 2012), (2013, 2017), (2018, 2023)]
    quarter_dfs = []
    for start_year, end_year in quarters:
        quarter_dfs.append(df[(df['year'] >= start_year) & (df['year'] <= end_year)])
    transfer_totals = []

    # calculate total transfer fees for each quarter and convert to billions of euros
    for dfs in quarter_dfs:
        total = dfs['market_value_in_eur'].sum(skipna = True)
        total /= 1000000000
        transfer_totals.append(total)

    # plot the bar graph for the transfer fees for each quarter
    positions1 = np.arange(len(quarters))
    plt.bar(positions1,transfer_totals, width=0.5, color='r')
    plt.xticks(positions1, ['2003-2007', '2008-2012', '2013-2017', '2018-2023'])
    plt.xlabel('5 Year Periods')
    plt.ylabel('Total fees (in billions of euros)')
    plt.title('Transfer fees in 5 year periods from 2003-2023')
    plt.show()

    # project future transfer fees using a non-linear regression model
    # add the projected data to the original data
    transfer_totals_f = list(transfer_totals)
    transfer_totals_f.extend([743.78289323, 842.4144779])
    quarters_f = list(quarters)
    quarters_f.extend([(2023, 2027), (2028, 2032)])

    # plot the bar graph for the projected transfer fees
    positions2 = np.arange(len(quarters_f))
    plt.bar(positions2,transfer_totals_f, width=0.5, color='b')
    plt.xticks(positions2, ['2003-2007', '2008-2012', '2013-2017', '2018-2023', '2023-2027', '2028-2032'])
    plt.xlabel('5 Year Periods')
    plt.ylabel('Total fees (in billions of euros)')
    plt.title('Transfer fees projections using Non-Linear regression : logistics Model')
    plt.show()

    

file_func()
growth_in_value()
transfer_over_time()
