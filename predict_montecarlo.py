import os, sys, pickle
import pandas as pd
import numpy as np
from datetime import datetime
import nfl_data_py as nfl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from optparse import OptionParser
from sklearn.decomposition import PCA

from numba import njit
from diskcache import Cache

# Define the cache directory
CACHE_DIR = "nfl_data_cache"
cache = Cache(CACHE_DIR)

pd.set_option('display.max_columns', None)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SEED = 42

models = {
    'Logistic Regression':          (LogisticRegression,{'max_iter': 1000}),
    'K-Nearest Neighbors':          (KNeighborsClassifier,{}),
    'Support Vector Machine':       (SVC,{}),
    'Decision Tree':                (DecisionTreeClassifier, {'random_state':SEED}),
    'Random Forest':                (RandomForestClassifier, {'n_estimators':500, 'random_state':SEED}),
    'Gradient Boosting':            (GradientBoostingClassifier,{}),
    'AdaBoost':                     (AdaBoostClassifier,{}),
    'Naive Bayes':                  (GaussianNB,{}),
    # 'XGBoost': XGBClassifier(),
    # 'LightGBM': lgb.LGBMClassifier(),
    'CatBoost':                     (CatBoostClassifier, {'learning_rate':0.1, 'iterations':100, 'depth':6, 'verbose':0}),
    'Extra Trees':                  (ExtraTreesClassifier,{}),
    'Neural Network':               (MLPClassifier, {'hidden_layer_sizes':(100, 50), 'max_iter':1000, 'random_state':SEED}),
}






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
        data_info_old = pickle.load(f)
    # verify that the data info is the same
    if data_info_old != data_info:
        return None

    # load data
    file_path = os.path.join(CACHE_DIR, f"fetched_data.pkl")
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

@cache.memoize()
def fetch_data_with_cache(start_year, end_year, end_week, season_type, current_year):
    all_pbp_data = []
    game_cols =['game_id', 'home_team', 'away_team', 'week', 'game_date', 'season_type', 'year']
    off_filter = {
                'game_id2': 'first', 
                'team': 'first', 
                'home_away':'first', 
                'drive':'max', 
                'touchdown':'sum',
                'field_goal_result':'sum', 
                'total_home_score': 'max', 
                'total_away_score': 'max',
                'play_type': 'max', 
                "yards_gained": 'sum', 
                'ydstogo': 'mean', 
                'half_seconds_remaining': 'mean', 
                'game_seconds_remaining': 'mean',
                'down': 'sum',
                'yardline_100': 'mean',
                'score_differential': 'mean',
                'ydsnet': 'sum',
                'year': 'first',
                'week': 'first',
                'result': 'first',
                
                  }
    def_filter = {'game_id2': 'first', 
                  'team': 'first', 
                  'home_away':'first', 
                  'drive':'max', 
                  'touchdown':'sum',
                  'field_goal_result':'sum', 
                  'total_home_score': 'max', 
                  'total_away_score': 'max',}
    stats_cols = ['drive_score_percentage', 'drive_score_percentage_def', 'score', 'def_score', 'previous_win', ]


    for season in range(start_year, end_year + 1):
        print(f"Fetching data for {season}...")
        # Fetch play-by-play data
        pbp_data = get_cached_data(season, "pbp")
        if pbp_data is None or (season == current_year):
            print(f"Cache miss.. Fetching play-by-play data for {season}...")
            try:
                pbp_data = nfl.import_pbp_data([season], include_participation=False)
            except:
                continue
            save_to_cache(pbp_data, season, "pbp")
        pbp_data['year'] = season
        if season == end_year:
            pbp_data = pbp_data[pbp_data['week'] <= end_week]
        all_pbp_data.append(pbp_data)


    # Combine all seasons' data into single DataFrames
    pbp_data = pd.concat(all_pbp_data, ignore_index=True)
    if season_type:
        pbp_data = pbp_data[pbp_data['season_type'] == season_type]
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

    # merge the offsense and defense data
    # final_summary = off_data.merge(def_data, how='inner', on=['game_id2', 'team'], suffixes=('_off', '_def'))
    # final_summary = final_summary.merge(static_data, how='left', left_on='game_id2', right_on='game_id')

    final_summary = off_data

    return final_summary, static_data, game_cols



def filter_stats(final_summary, stats, alpha = .05, season_discount=.9):
    final_summary = final_summary.copy()
    filter_size = 20
    years=final_summary['year']
    year_scaled = (years - years.iloc[0]+2)
    discount_weights = np.exp(-1 * year_scaled * np.log(season_discount))
    # precompute the filter sum of the weights so that we can divide by it later
    #weight_sums = discount_weights.ewm(alpha=alpha).mean()
    filt = np.exp(np.array(range(filter_size)) * np.log(1-alpha))
    weight_sums = np.convolve(discount_weights, filt, 'same')
    final_summary[stats] = final_summary[stats].mul(discount_weights, axis=0)

    # apply a weighted average filter to final_summary 
    # to get the stats that we want
    # Loop through each team separately
    for team in final_summary['team'].unique():
        team_data = final_summary[final_summary['team'] == team]

        index = team_data.index
        temp = team_data[stats].apply(lambda x: np.convolve(x, filt, "full")).iloc[:(-1 * filter_size + 1)]
        temp.index = index
        team_data.loc[:, stats] = temp
        # put team_data back into final_summary
        final_summary[final_summary['team'] == team] = team_data

    final_summary[stats] = final_summary[stats].div(weight_sums, axis=0)

    # remove the first season
    first_season = final_summary['year'][0]
    final_summary = final_summary.loc[final_summary['year'] != first_season]
    # Display the first few rows of the DataFrame
    return final_summary



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

def find_correlation(X, Y):
    #stat_map = {s:'mean' for s in stats}
    #correlation = pbp_data.groupby(['team', 'year']).agg(stat_map).reset_index()[stats]
    #stat_data = pbp_data.drop('result', axis=1).shift(1).dropna().reset_index()
    correlation = pd.concat([X, Y], axis=1)
    # Calculate the correlation between the two variables
    correlation = correlation.corr()
    # Display the correlation matrix
    print(correlation)

def form_vectors(final_summary, stats):
    # get all of the home_away rows that equal home
    home = final_summary[final_summary['home_away'] == 'home']
    # get all of the home_away rows that equal away
    away = final_summary[final_summary['home_away'] == 'away']
    vectors = home.merge(away, how='inner', on='game_id', suffixes=('_home', '_away'))
    new_stats = []
    for stat in stats:
        new_stats.append(stat+ '_home')
        new_stats.append(stat+ '_away')

    vectors = vectors.rename(columns={'result_home': 'result'})
    return vectors, new_stats



def get_xy(vectors, stats):
    X = vectors[stats]
    y = vectors['result']
    return X, y

def get_records(data, stats_cols, alpha, season_discount, options):
    data = data.copy()
    data_filtered = filter_stats(data, stats_cols, alpha=alpha, season_discount=season_discount)

    vectors, new_stats = form_vectors(data_filtered, stats_cols)
    vectors = vectors.dropna()
    # Split the data into training and testing sets

    X, Y = get_xy(vectors, new_stats)
    return X, Y, vectors




def play_game(data, home, away):
    
    data[['weight', 'distance']] = 0.0



    normalization_factors = pd.Series({
        "yardline_100":100.0, 
        "ydstogo":10.0, 
        "down": 1, 
        "score_differential": 30.0, 
        'half_seconds_remaining': 30 * 60, 
        'game_seconds_remaining': 60 * 60,
        })
    
    required_columns = ['team', 'play_type', 'yards_gained', 'ydsnet', 'down', 'ydstogo', 'yardline_100', 'score_differential', 'half_seconds_remaining', 'game_seconds_remaining']

    stats = {}
    for team in [home, away]:
        home_data  = data[data['team'] == team].loc[:, required_columns].copy()
        
        home_data = home_data[home_data['play_type'].isin(['run', 'pass', 'punt', 'field_goal'])]
        home_data[normalization_factors.keys()] = home_data[normalization_factors.index].div(normalization_factors)
        home_data = home_data.dropna()
        home_stats = {}
        for i in range(4):
            home_stats[i+1] = home_data[ home_data['down'] == i+1]
            if i == 3:
                home_stats['no_fg'] = home_stats[i][home_stats[i]['play_type'] != 'field_goal']
                home_stats['no_punt'] = home_stats[i][home_stats[i]['play_type'] != 'punt']
        stats[team] = home_stats
    
    situation = {
                'down': 1, 
                'ydstogo': 10, 
                'yardline_100': 75,
                'score_differential': 0,
                'half_seconds_remaining': 30 * 60,
                'game_seconds_remaining': 60  * 60,
    }
    situation = pd.Series(situation)
    score  = {home: 0, away: 0}

    


    first_possession = home if np.random.rand() > 0.5 else away
    possession = first_possession

    while situation['game_seconds_remaining'] > 0:
        # print("Situation:", situation.to_dict())
        stat_records = stats[possession][situation['down']]
        if situation['down'] == 4:
            if situation['yardline_100'] > 45:
                stat_records = stats[possession]['no_fg']
            elif situation['yardline_100'] < 20:
                stat_records = stats[possession]['no_punt']

        norm_situation = situation / normalization_factors
        distances = stat_records[situation.keys()].sub(norm_situation).pow(2).sum(axis=1)
        # find what the play is pass, run, field goal, punt given the situation
        # the situation is down, yards to go, and field position
        # if it's the first second or third down remove punt and field goal plays


        # select a random play_type weighted by distance
        weights = np.exp(-1 * distances)
        random_row = stat_records.sample(n=1, weights=weights).iloc[0]
        play_type = random_row['play_type']

        scored = False
        change_possession = False
        first_down = False
        # how many yards given the situation.  The situlation is down, yards to go, and field position
        if play_type in ['run', 'pass']:
            yards_gained = int(random_row['yards_gained'])
            situation['yardline_100'] -= yards_gained
            situation['ydstogo'] -= yards_gained
            if situation['ydstogo'] <= 0:
                first_down = True
            if situation['yardline_100'] <= 0:
                situation['yardline_100'] = 75
                score[possession] += 7
                scored = True
        elif play_type == 'punt':
            punt_distance = random_row['ydsnet']
            situation['yardline_100'] -= punt_distance
            if situation['yardline_100'] <= 0:
                situation['yardline_100'] = 25
            change_possession = True
        elif play_type == 'field_goal':
            # if the field goal is good, add 3 points to the score and change possession
            if np.random.rand() < 0.85: # assume 85% chance of making a field goal
                score[possession] += 3
                scored = True
            else:
                change_possession = True
        

        # print(f"Outcome: {play_type}, Yards Gained: {yards_gained if play_type in ['run', 'pass'] else 'N/A'}, Score: {score}")

        # how much time does the play take including the time it takes to huddle and line up
        # update the situation based on the play that was run
        # if the play was a touchdown, update the score and reset the situation
        # if the play was a field goal, update the score and reset the situation   
        play_time = np.random.randint(20, 40)
        situation['game_seconds_remaining'] -= play_time
        situation['half_seconds_remaining'] -= play_time

        if situation['down'] == 4 and situation['ydstogo'] > 0 and not first_down or change_possession:
            situation['yardline_100'] = 100 - situation['yardline_100']
            situation['down'] = 1
            situation['ydstogo'] = 10
            situation['score_differential'] = score[home] - score[away] if possession == home else score[away] - score[home]
            possession = home if possession == away else away
        elif scored:
            situation['down'] = 1
            situation['ydstogo'] = 10
            situation['yardline_100'] = 75
            situation['score_differential'] = score[home] - score[away] if possession == home else score[away] - score[home]
            possession = home if possession == away else away
        elif first_down:
            situation['down'] = 1
            situation['ydstogo'] = min(10, 100 - situation['yardline_100'])
        else:
            situation['down'] += 1
        
        if situation['half_seconds_remaining'] <= 0:
            situation['half_seconds_remaining'] = 30 * 60
            # switch sides
            situation['yardline_100'] = 100 - situation['yardline_100']
            possession = home if first_possession == away else away
            change_possession = False
            first_down = False
            situation['down'] = 1
            situation['ydstogo'] = 10
    
    return score


def monte_carlo_simulation(data, home, away, n=1000, run_parallel=True):
    home_wins = 0
    away_wins = 0
    if run_parallel:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1)(delayed(play_game)(data, home, away) for _ in range(n))
        for score in results:
            if score[home] > score[away]:
                home_wins += 1
            elif score[away] > score[home]:
                away_wins += 1
    else:
        for i in range(n):
            print(f"Simulation {i+1}/{n}")
            score = play_game(data, home, away)
            if score[home] > score[away]:
                home_wins += 1
            elif score[away] > score[home]:
                away_wins += 1
    print(f"After {n} simulations:")
    print(f"{home} wins: {home_wins} ({home_wins/n*100:.2f}%)")
    print(f"{away} wins: {away_wins} ({away_wins/n*100:.2f}%)")
    print(f"Ties: {n - home_wins - away_wins} ({(n - home_wins - away_wins)/n*100:.2f}%)")
    return home_wins/n, away_wins/n, (n - home_wins - away_wins)/n


def backtest_monte(year, go_back = 3):
    # pull all data for the given year.  Predict each week using all previous data

    data, game_data, game_cols = fetch_data_with_cache(year-go_back, year , 17, 'REG', 2025)

    results = data.groupby('game_id').max(['total_home_score', 'total_away_score'])
    results.drop(['week', 'year'], axis=1, inplace=True)
    game_data = game_data.merge(results, how='left', on='game_id')

    correct = 0
    total = 0
    for week in range(1, 18):
        # get all data up to the given week
        # find all of the matchups for the given week
        matchups = game_data[(game_data['year'] == year) & (game_data['week'] == week)]
        week_data = data[(data['year'] < year) | ((data['year'] == year) & (data['week'] < week))]
        print(f"Predicting week {week} of {year} with {len(matchups)} games of data")
        for i, row in matchups.iterrows():
            print(f"Predicting {row['away_team']} at {row['home_team']}")
            # get all data up to the given week
            
            home, away, tie = monte_carlo_simulation(week_data,  row['home_team'], row['away_team'], n=100, run_parallel=True)
            print(f"Prediction: {row['home_team']} win probability: {home:.2f}, {row['away_team']} win probability: {away:.2f}, Tie probability: {tie:.2f}")
            winner = row['home_team'] if home > away else row['away_team'] if away > home else 'Tie'

            actual = row['home_team'] if row['total_home_score'] > row['total_away_score'] else row['away_team']
            if winner == actual:
                correct += 1
            total += 1

    print(f"Backtest accuracy: {correct}/{total} = {correct/total*100:.2f}%")

def main():
    parser = OptionParser()
    parser.add_option("-s", "--start_year", dest="start_year", type="int", default=2009,
                        help="The first year of data to fetch (default: 2009)")
    parser.add_option("-e", "--end_year", dest="end_year", type="int", default=None,
                        help="The last year of data to fetch (default: 2023)")
    parser.add_option("-w", "--end_week", dest="end_week", type="int", default=101,
                        help="The last week of the season to fetch (default: 17)")
    parser.add_option("-t", "--season_type", dest="season_type", default=None,
                        help="Restrict the data to a specific season type (e.g., 'REG', 'POST')")
    parser.add_option("-m", "--model", dest="model", default="Logistic Regression",
                        help="The model to use for prediction (default: RandomForest) one of: %s" % ", ".join(models.keys()))
    parser.add_option("--eval_model", dest="eval_model", default=False, action="store_true",
                    help="Run the data through a bunch of models and spit out stats")
    parser.add_option("--predict_year", dest="predict_year", default=None, type='int',
                    help="Predict the outcome of games for the given year.  Both this and predict_week must be set to predict games")
    parser.add_option("--predict_week", dest="predict_week", default=None, type='int',
                    help="Predict the outcome of games for the given week.  Both this and predict_week must be set to predict games")
    parser.add_option("-f","--force_data_gather", dest="force_data_gather", default=False, action="store_true",
                    help="Force the data to be gathered again even if it is in the cache.")
    parser.add_option("--find_filter_params", dest="find_filter_params", default=False, action="store_true",
                help="Force the data to be gathered again even if it is in the cache.")
    parser.add_option("--check_season", dest="check_season", default=False, action="store_true",
                help="Check to see how we are doing within the season.")
    
    (options, args) = parser.parse_args()

    # Get the current date and time
    now = datetime.now()

    # Extract the current month and year
    current_month = now.month
    current_year = now.year
    if current_month < 6:
        # need the nfl seasonal year
        current_year -= 1
    if options.end_year == None:
        options.end_year = current_year

    if options.predict_year and options.predict_week: # if we are predicting for a given year and week, don't get any data grater than that.
        options.end_year = int(options.predict_year)
    
    if options.predict_week:
        options.end_week = options.predict_week

    # Fetch the data with caching
    data, stats_cols, game_cols = fetch_data_with_cache(options.start_year, options.end_year, options.end_week, options.season_type, current_year)

    alpha = .32
    season_discount = .57

    print('Alpha: %f' % alpha)
    print('Season discount: %f' % season_discount)


    # home, away, tie = monte_carlo_simulation(data,  'NE', 'PIT',)
    # print(f"NE win probability: {home:.2f}, NYJ win probability: {away:.2f}, Tie probability: {tie:.2f}")
    backtest_monte(2023, go_back=3)



if __name__ == "__main__":
    main()



    # print_correlation(data, 'drive_score_percentage')
    # print_correlation(data, 'drive_score_percentage_def')
    # print_correlation(data, 'score')
    # print_correlation(data, 'def_score')
    # sys.exit(0)


