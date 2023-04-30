import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

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
    df = dataframes[6]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    avg_list = []
    count_list = []
    g = df.groupby('year')
    for y, y_df in g:
        f = y_df.market_value_in_eur
        avg_list.append(f.mean() / 1000000)
        count_list.append(f.count())
    # Plotting the line graph
    plt.plot(g.groups.keys(), avg_list, 'o-')
    plt.xlabel('Year')
    plt.ylabel('Average Market Value (Millions of Euros)')
    plt.title('Change in Market Values of Players over Time')
    plt.show()

    # Plotting the line graph
    plt.plot(g.groups.keys(), count_list, 'o-')
    plt.xlabel('Year')
    plt.ylabel('Number of Entries')
    plt.title('Distribution of Number of Entries per Year')
    plt.show()

    # As my comparision compare avg values over the years, more of the
    # recent years have higher number of entries, the data is skewed slightly
    # Therefore for comaprison, this code is taken from David Coxon as credited

    player_valuations_df = df[(df.year > 2004) & (df.year < 2023)]
    plt.figure(figsize=(10, 5))
    plt.scatter(player_valuations_df['datetime'], y=player_valuations_df['market_value_in_eur'] / 1000000, c='b',
                alpha=0.15)
    plt.xlabel('Date');
    plt.ylabel('Valuation in million euros')
    plt.title('Player valuations over time')
    plt.show()


def value_over_time():
    # load the dataset and create year column
    df = dataframes[6]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year

    # create list of quarters and dataframes for each quarter
    quarters = [(2003, 2007), (2008, 2012), (2013, 2017), (2018, 2023)]
    quarter_dfs = []
    for start_year, end_year in quarters:
        quarter_dfs.append(df[(df['year'] >= start_year) & (df['year'] <= end_year)])
    value_totals = []

    # calculate total transfer fees for each quarter and convert to billions of euros
    for dfs in quarter_dfs:
        total = dfs['market_value_in_eur'].sum(skipna = True)
        total /= 1000000000
        value_totals.append(total)

    # plot the bar graph for the transfer fees for each quarter
    positions1 = np.arange(len(quarters))
    plt.bar(positions1,value_totals, width=0.5, color='r')
    plt.xticks(positions1, ['2003-2007', '2008-2012', '2013-2017', '2018-2023'])
    plt.xlabel('5 Year Periods')
    plt.ylabel('Total Market values (in billions of euros)')
    plt.title('Market values in 5 year periods from 2003-2023')
    plt.show()

    # project future transfer fees using a non-linear regression model
    # add the projected data to the original data
    value_totals_f = list(value_totals)
    value_totals_f.extend([743.78289323, 842.4144779])
    quarters_f = list(quarters)
    quarters_f.extend([(2023, 2027), (2028, 2032)])

    # plot the bar graph for the projected transfer fees
    positions2 = np.arange(len(quarters_f))
    plt.bar(positions2,value_totals_f, width=0.5, color='b')
    plt.xticks(positions2, ['2003-2007', '2008-2012', '2013-2017', '2018-2023', '2023-2027', '2028-2032'])
    plt.xlabel('5 Year Periods')
    plt.ylabel('Total Market values (in billions of euros)')
    plt.title('Market Values projections using Non-Linear regression : logistics Model')
    plt.show()

#def value_over_time_by_position():

def total_goals():
    df = dataframes[5]
    h_game = df.groupby('home_club_id')
    a_game = df.groupby('away_club_id')

    h_goals = {}
    a_goals = {}

    for id, group in h_game:
        h_goals[id] = group['home_club_goals'].sum()
    for name, group in a_game:
        a_goals[id] = group['away_club_goals'].sum()

    total_goals = {}
    for name in set(h_goals.keys()) | set(a_goals.keys()):
        total_goals[name] = h_goals.get(name, 0) + a_goals.get(name, 0)
    # print(total_goals)

    res = dict(sorted(total_goals.items(), key=itemgetter(1), reverse=True)[:10])
    most_goals = {}
    for k, v in res.items():
        d = dataframes[2][dataframes[2].club_id == k]
        most_goals[d['name'].iloc[0]] = v
    x_data = list(most_goals.keys())
    y_data = list(most_goals.values())

    # Create the bar chart
    plt.figure(figsize=(22, 8))
    plt.bar(x_data, y_data)

    # Set the title and axis labels
    plt.title("Top 10 clubs with most goals over the last 10 years")
    plt.xlabel("Club name")
    plt.ylabel("Total goals")

    # Show the plot
    plt.show()


file_func()
growth_in_value()
value_over_time()
total_goals()


'''
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
'''
