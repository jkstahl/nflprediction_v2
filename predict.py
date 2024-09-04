import os, sys, pickle
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Define the cache directory
CACHE_DIR = "nfl_data_cache"

# Ensure the cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cached_data(season, data_type, pickleit = False):
    """Load cached data for a given season and data type (e.g., 'pbp', 'drives')."""
    file_path = os.path.join(CACHE_DIR, f"{data_type}_season_{season}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return None

def save_to_cache(data, season, data_type, pickleit = False):
    """Save fetched data to the cache."""
    file_path = os.path.join(CACHE_DIR, f"{data_type}_season_{season}.csv")
    data.to_csv(file_path, index=False)

def save_to_pickle_cache(data, data_info):
    # save data info
    file_path = os.path.join(CACHE_DIR, f"fetched_data_info.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(data_info, f)
    # save data
    file_path = os.path.join(CACHE_DIR, f"fetched_data.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle_cache(data_info):
    # load data info
    file_path = os.path.join(CACHE_DIR, f"fetched_data_info.pkl")
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rb') as f:
        data_info = pickle.load(f)
    # verify that the data info is the same
    if data_info != data_info:
        return None

    # load data
    file_path = os.path.join(CACHE_DIR, f"fetched_data.pkl")
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def fetch_data_with_cache(start_year, end_year, end_week=17):
    all_pbp_data = []
    game_cols =['game_id', 'home_team', 'away_team', 'total_home_score', 'total_away_score', 'week', 'game_date', 'season_type', 'year']
    off_filter = {'game_id2': 'first', 'team': 'first', 'home_away':'first', 'drive':'max', 'touchdown':'sum','field_goal_result':'sum', 'total_home_score': 'max', 'total_away_score': 'max',}
    def_filter = {'game_id2': 'first', 'team': 'first', 'home_away':'first', 'drive':'max', 'touchdown':'sum','field_goal_result':'sum', 'total_home_score': 'max', 'total_away_score': 'max',}
    stats_cols = ['drive_score_percentage', 'drive_score_percentage_def', 'score', 'def_score', 'previous_win', ]

    data_info = (start_year, end_year, end_week,  game_cols, off_filter, def_filter, stats_cols,)
    # try to load the data from file cache
    data = load_from_pickle_cache(data_info)

    if data is not None:
        return data, stats_cols, game_cols

    for season in range(start_year, end_year + 1):
        # Fetch play-by-play data
        pbp_data = get_cached_data(season, "pbp")
        if pbp_data is None:
            pbp_data = nfl.import_pbp_data([season])
            save_to_cache(pbp_data, season, "pbp")
        pbp_data['year'] = season
        all_pbp_data.append(pbp_data)

        # # Fetch drive data
        # drive_data = get_cached_data(season, "drives")
        # if drive_data is None:
        #     drive_data = nfl.import_drives([season])
        #     save_to_cache(drive_data, season, "drives")
        # all_drive_data.append(drive_data)

    # Combine all seasons' data into single DataFrames
    pbp_data = pd.concat(all_pbp_data, ignore_index=True)
    pbp_data['game_id2'] = pbp_data['game_id']

    pbp_data['touchdown'] = pbp_data['pass_touchdown'] + pbp_data['rush_touchdown']
    pbp_data['team'] = pbp_data['posteam']

    # change the column name posteam_type to home_away
    pbp_data = pbp_data.rename(columns={'posteam_type': 'home_away'})

    # get satic data columns

    static_data = pbp_data[game_cols].drop_duplicates('game_id')

    # filter only the cols that we need
    off_data = pbp_data[list(off_filter.keys()) + ['game_id', 'posteam', 'defteam']]
    def_data = pbp_data[list(def_filter.keys()) + ['game_id', 'posteam', 'defteam']]
    def_data['team'] = def_data['defteam']

    # generate field goals for off and def
    off_data['field_goal_result'] = off_data['field_goal_result'].apply(lambda x: 1.0 if x == 'made' else 0.0 ).fillna(0.0)
    def_data['field_goal_result'] = def_data['field_goal_result'].apply(lambda x: 1.0 if x == 'made' else 0.0 ).fillna(0.0)

    # regroup the data to per game and generate the 
    off_data = off_data.groupby(['game_id', 'posteam']).agg(off_filter).reset_index()
    def_data = def_data.groupby(['game_id', 'defteam']).agg(def_filter).reset_index()

    # merge the offsense and defense data
    final_summary = off_data.merge(def_data, how='inner', on=['game_id2', 'team'], suffixes=('_off', '_def'))
    final_summary = final_summary.merge(static_data, how='left', left_on='game_id2', right_on='game_id')

    # rename home_away_off to home_away
    final_summary = final_summary.rename(columns={'home_away_off': 'home_away'})

    # create a new column result which is 1.0 if the team won and 0.0 if the team lost
    # final_summary['home_team'] = final_summary['home_team_off']
    # final_summary['away_team'] = final_summary['away_team_off']
    #final_summary['team'] = final_summary['team_off']
    # Compute some stats we need
    final_summary['score'] = final_summary.apply(lambda x: x['total_home_score_off'] if x['home_team'] == x['team'] else x['total_away_score_off'], axis=1)
    final_summary['def_score'] = final_summary.apply(lambda x: x['total_away_score_off'] if x['home_team'] == x['team'] else x['total_home_score_off'], axis=1)
    final_summary['result'] = final_summary.apply(lambda x: float(x['home_team'] == x['team']) if x['total_home_score_off'] > x['total_away_score_off'] else float(x['away_team'] == x['team']), axis=1)
    final_summary['drive_score_percentage'] = (final_summary['touchdown_off'] + final_summary['field_goal_result_off']) / final_summary['drive_off']
    final_summary['drive_score_percentage_def'] = 1.0 - (final_summary['touchdown_def'] + final_summary['field_goal_result_def']) / final_summary['drive_def']
    # change game_id2 to game_id
    #final_summary = final_summary.rename(columns={'game_id2': 'game_id'})

    # need to get the previous win because the current win is what we are trying to predict and so won't be available
    final_summary['previous_win'] = final_summary.groupby('team')['result'].shift(1).dropna()

    results = final_summary[['game_id', 'team', 'result']]
    final_summary = final_summary[stats_cols + game_cols + ['team', 'home_away', 'result']]

    # save all data to pickle cache file
    save_to_pickle_cache(final_summary, data_info)

    return final_summary, stats_cols, game_cols


def filter_stats(final_summary, stats):
    # apply a weighted average filter to final_summary 
    # to get the stats that we want
    # Loop through each team separately
    for team in final_summary['team'].unique():
        team_data = final_summary[final_summary['team'] == team]
        # run a filter on the data that weights the latest games more
        # and returns the stats that we want
        team_data[stats] = team_data[stats].rolling(window=10).mean()
        # take a windowed average of the last 10 games
        # put team_data back into final_summary
        final_summary[final_summary['team'] == team] = team_data

    # Display the first few rows of the DataFrame
    return final_summary.dropna()



def print_correlation(final_summary, stat):
    # using matplotlib to plot drive_score_percentage vs score
    import matplotlib.pyplot as plt
    correlation = final_summary.groupby(['team', 'year']).agg({stat: 'mean', 'previous_win': 'mean'}).reset_index()
    plt.scatter(correlation[stat], correlation['previous_win'])
    stat_label =stat.replace('_', ' ').title()
    plt.xlabel(stat_label)
    plt.ylabel('Win Rate')
    plt.title('%s vs. win rate' % stat_label)
    plt.show()

def find_correlation(pbp_data, stats,):
    stat_map = {s:'mean' for s in stats}
    correlation = pbp_data.groupby(['team', 'year']).agg(stat_map).reset_index()[stats]
    # Calculate the correlation between the two variables
    correlation = correlation.corr()
    # Display the correlation matrix
    print(correlation)

def form_vectors(final_summary, stats):
    # get all of the home_away rows that equal home
    home = final_summary[final_summary['home_away'] == 'home']
    # get all of the home_away rows that equal away
    away = final_summary[final_summary['home_away'] == 'away']
    vectors = home.merge(away, how='inner', on='game_id')
    new_stats = []
    for stat in stats:
        new_stats.append(stat+ '_x')
        new_stats.append(stat+ '_y')
    results = vectors[['game_id', 'result_x']]
    return vectors, new_stats, results

if __name__ == "__main__":
    # Example usage
    start_year = 2009
    end_year = 2021

    # Fetch the data with caching
    data, stats_cols, game_cols = fetch_data_with_cache(start_year, end_year)
    data_save = data.copy()

    find_correlation(data, stats_cols)

    data_filtered = filter_stats(data, stats_cols)

    vectors, new_stats, output = form_vectors(data_filtered, stats_cols)
    X = vectors[new_stats]
    y = output['result_x']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # print_correlation(data, 'drive_score_percentage')
    # print_correlation(data, 'drive_score_percentage_def')
    # print_correlation(data, 'score')
    # print_correlation(data, 'def_score')
    # sys.exit(0)


