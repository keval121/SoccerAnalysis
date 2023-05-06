import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from operator import itemgetter
from datetime import datetime

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

def value_over_time_by_position():
    df = dataframes[6]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    sf = dataframes[7]
    pos_dict = {}
    for id in df['player_id'].unique():
        pos_dict[id] = sf.loc[sf['player_id'] == id, 'position'].values[0]
    df['position'] = df['player_id'].map(pos_dict)

    quarters = [(2003, 2007), (2008, 2012), (2013, 2017), (2018, 2023)]
    median_yr = []
    for start, end in quarters:
        median_yr.append(int(start + end / 2))

    positions = ['Goalkeeper', 'Defender', 'Midfield', 'Attack']
    datasets = {}

    for position in positions:
        datasets[position] = {}
        i = 0
        for start_year, end_year in quarters:
            rows = df[(df['year'] >= start_year) & (df['year'] <= end_year) & (df['position'] == position)]
            datasets[position][median_yr[i]] = rows
            i += 1

    datasets_tot = {}
    for position in positions:
        datasets_tot[position] = {}
        total = 0
        for myr in median_yr:
            total = datasets[position][myr]['market_value_in_eur'].sum(skipna = True)
            total /= 100000000
            datasets_tot[position][myr] = total

    # Set the color for each position
    colors = {'Goalkeeper': 'green', 'Defender': 'yellow', 'Midfield': 'blue', 'Attack': 'red'}

    # Prepare data for each position
    data = {}
    for position in positions:
        data[position] = [datasets_tot[position][myr] for myr in median_yr]

    # Prepare the x-ticks labels
    x_ticks = [f"{start}-{end}" for start, end in quarters]

    # Set the width of each bar
    bar_width = 0.2

    # Prepare the x-ticks positions for each section
    x_pos = np.arange(len(x_ticks))

    # Plot the graph
    fig, ax = plt.subplots(figsize=(15, 8))
    for i, position in enumerate(positions):
        ax.bar(x_pos + (i - 1.5) * bar_width, data[position], width=bar_width, color=colors[position], label=position)

    # Set the x-ticks and labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel('5 year periods')

    # Set the y-label and title
    ax.set_ylabel('Market value in Billions of Euros')
    ax.set_title('Market value by position and 5 year periods')

    # Set the legend
    ax.legend()

    # Show the graph
    plt.show()

def player_value_by_age():
    df = dataframes[7]
    birth_day = pd.to_datetime(df['date_of_birth'])
    today = datetime.today()
    df['age'] = (today - birth_day).dt.total_seconds() / (365.2425 * 24 * 3600)
    df['age'] = df['age'].fillna(0).astype(int)
    current = df[df['last_season'] == 2022]
    retired = df[df['last_season'] < 2022]

    periods = [(16, 20), (21, 25), (26, 30), (31, 35), (36, 100)]
    age_groups_dfs = []
    for start, end in periods:
        agdfs = current[(current['age'] >= start) & (current['age'] <= end)]
        age_groups_dfs.append(agdfs)

    total_value_by_age = []
    for agdfs in age_groups_dfs:
        total = agdfs['market_value_in_eur'].sum(skipna=True) / 1000000000
        total_value_by_age.append(total)

    retired_years = retired.groupby('last_season')
    retired_dict = {}
    for year, data in retired_years:
        retired_dict[year] = data['market_value_in_eur']

    # flatten the values in retired_dict
    from matplotlib import ticker

    values = [val for sublist in retired_dict.values() for val in sublist]

    # create a list of corresponding years
    years = []
    for year in retired_dict:
        years += [year] * len(retired_dict[year])

    # create scatter plot with flattened data
    plt.scatter(years, values, s=100, alpha=0.7)
    plt.xlabel('Year')
    plt.ylabel('Market Value at the time of retirement (M€)')
    plt.ticklabel_format(style='plain', axis='y')
    plt.gca().get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # create a bar plot for total value by age
    plt.figure(figsize=(10, 6))
    x_labels = ['Teen (16-20)', 'Young (21-25)', 'Prime (26-30)', 'Veterans (31-35)', 'Old (36+)']
    sns.barplot(x=x_labels, y=total_value_by_age)
    plt.xlabel('Age Group')
    plt.ylabel('Total Market Value (B€)')
    plt.title('Total Market Value of Active Players by Age Group')
    plt.show()


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
    plt.figure(figsize=(15, 6))
    plt.bar(x_data, y_data)

    # Set the title and axis labels
    plt.title("Top 10 clubs with most goals over the last 10 years")
    plt.xlabel("Club name")
    plt.ylabel("Total goals")
    plt.xticks(rotation = 90)

    # Show the plot
    plt.show()

def top_players_and_stats():
    df = dataframes[6]
    af = dataframes[7]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    sf = dataframes[0]

    # Keep only the most recent year for each player
    df_recent = df.loc[df.groupby('player_id')['year'].idxmax()]
    # Sort by market value and keep top 10 rows
    df_cur15 = df_recent.sort_values(by='market_value_in_eur', ascending=False).head(15)
    # Print the top 10 players
    names_dict = {}
    for id in df_cur15['player_id']:
        row = af[af['player_id'] == id]
        names_dict[id] = row['name'].values[0]
    df_cur15['name'] = df_cur15['player_id'].map(names_dict)
    af_copy = af[af['highest_market_value_in_eur'] == af['highest_market_value_in_eur']].copy()
    af_copy.sort_values(by='highest_market_value_in_eur', ascending=False, inplace=True)
    af_top15 = af_copy[0:15]

    # Add total goals and assists columns to df_cur10
    copy_sf_df = sf[sf['player_id'].isin(df_cur15['player_id'])]
    copy_sf_df = copy_sf_df.groupby('player_id').agg({'goals': 'sum', 'assists': 'sum'}).reset_index()
    df_cur15 = pd.merge(df_cur15, copy_sf_df, on='player_id', how='left')

    # Add total goals and assists columns to af_top10
    copy_sf_af = sf[sf['player_id'].isin(af_top15['player_id'])]
    copy_sf_af = copy_sf_af.groupby('player_id').agg({'goals': 'sum', 'assists': 'sum'}).reset_index()
    af_top15 = pd.merge(af_top15, copy_sf_af, on='player_id', how='left')

    fig, axs = plt.subplots(ncols=2, figsize=(15, 6))
    sns.barplot(x='market_value_in_eur', y='name', data=df_cur15, ax=axs[0])
    axs[0].set_title('Current Most Valued Players')
    axs[0].set_xlabel('Market Value (in Millions Euro)')
    sns.barplot(x='highest_market_value_in_eur', y='name', data=af_top15, ax=axs[1])
    axs[1].set_title('Most Valued Players of All Time')
    axs[1].set_xlabel('Market Value (in Millions Euro)')
    plt.subplots_adjust(wspace=0.5)
    plt.show()

    fig, axs = plt.subplots(ncols=2, figsize=(25, 7))
    sns.lineplot(x='name', y='goals', data=df_cur15, ax=axs[0], color='red', label='Goals')
    sns.lineplot(x='name', y='assists', data=df_cur15, ax=axs[0], color='blue', label='Assists')
    axs[0].set_title('Goals and Assists - Current Top Value Players')
    axs[0].set_xlabel('Player Name')
    axs[0].set_ylabel('Returns')
    axs[0].tick_params(axis='x', labelrotation=90)
    sns.lineplot(x='name', y='goals', data=af_top15, ax=axs[1], color='red', label='Goals')
    sns.lineplot(x='name', y='assists', data=af_top15, ax=axs[1], color='blue', label='Assists')
    axs[1].set_title('Goals and Assists - Highest Valued Players All Time')
    axs[1].set_xlabel('Player Name')
    axs[1].set_ylabel('Returns')
    axs[1].tick_params(axis='x', labelrotation=90)
    axs[1].legend()
    plt.subplots_adjust(wspace=0.5)
    plt.show()



file_func()
growth_in_value()
value_over_time()
value_over_time_by_position()
player_value_by_age()
total_goals()
top_players_and_stats()


'''
This function was created before I realized I misinterpreted the data
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
