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

pd.set_option('display.max_columns', None)



# from xgboost import XGBClassifier


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

def fetch_data_with_cache(start_year, end_year, end_week, options, current_year):
    all_pbp_data = []
    game_cols =['game_id', 'home_team', 'away_team', 'total_home_score', 'total_away_score', 'week', 'game_date', 'season_type', 'year']
    off_filter = {'game_id2': 'first', 'team': 'first', 'home_away':'first', 'drive':'max', 'touchdown':'sum','field_goal_result':'sum', 'total_home_score': 'max', 'total_away_score': 'max',}
    def_filter = {'game_id2': 'first', 'team': 'first', 'home_away':'first', 'drive':'max', 'touchdown':'sum','field_goal_result':'sum', 'total_home_score': 'max', 'total_away_score': 'max',}
    stats_cols = ['drive_score_percentage', 'drive_score_percentage_def', 'score', 'def_score', 'previous_win', ]

    data_info = (start_year, end_year, end_week,  game_cols, off_filter, def_filter, stats_cols,)
    # try to load the data from file cache
    data = load_from_pickle_cache(data_info)

    if data is not None and not options.force_data_gather:
        return data, stats_cols, game_cols

    for season in range(start_year, end_year + 1):
        print(f"Fetching data for {season}...")
        # Fetch play-by-play data
        pbp_data = get_cached_data(season, "pbp")
        if pbp_data is None:
            print(f"Cache miss.. Fetching play-by-play data for {season}...")
            try:
                pbp_data = nfl.import_pbp_data([season])
            except:
                continue
            save_to_cache(pbp_data, season, "pbp")
        pbp_data['year'] = season
        if season == end_year:
            pbp_data = pbp_data[pbp_data['week'] <= end_week]
        all_pbp_data.append(pbp_data)

        # # Fetch drive data
        # drive_data = get_cached_data(season, "drives")
        # if drive_data is None:
        #     drive_data = nfl.import_drives([season])
        #     save_to_cache(drive_data, season, "drives")
        # all_drive_data.append(drive_data)



    # Combine all seasons' data into single DataFrames
    pbp_data = pd.concat(all_pbp_data, ignore_index=True)
    if options.season_type:
        pbp_data = pbp_data[pbp_data['season_type'] == options.season_type]
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
    final_summary['previous_win'] = final_summary['result']#final_summary.groupby('team')['result'].shift(1).dropna()

    # results = final_summary[['game_id', 'team', 'result']]
    final_summary = final_summary[stats_cols + game_cols + ['team', 'home_away', 'result']]

    cols = final_summary.columns
    schedule_data = nfl.import_schedules([current_year])
    schedule_data['year'] = current_year
    schedule_data['season_type'] = 'REG'
    schedule_data = schedule_data.loc[schedule_data.index.repeat(2)].reset_index(drop=True)
    schedule_data.loc[schedule_data.index % 2 == 0, 'team'] = schedule_data.loc[schedule_data.index % 2 == 0, 'home_team']
    schedule_data.loc[schedule_data.index % 2 == 0, 'home_away'] = 'home'
    schedule_data.loc[schedule_data.index % 2 == 1, 'team'] = schedule_data.loc[schedule_data.index % 2 == 1, 'away_team']
    schedule_data.loc[schedule_data.index % 2 == 1, 'home_away'] = 'away'
    final_summary = pd.concat([final_summary, schedule_data])[cols].reset_index(drop=True)

    # This part shifts the game rows so that the result of the game is aligned with the stats of the previous game
    # Loop through each team separately and shift the result column back one.
    for team in final_summary['team'].unique():
        team_data = final_summary[final_summary['team'] == team]
        # team_data['result'] = team_data['result'].shift(-1).dropna()
        team_data.loc[:, stats_cols] = team_data[stats_cols].shift(1)
        final_summary[final_summary['team'] == team] = team_data


    # save all data to pickle cache file
    save_to_pickle_cache(final_summary, data_info)

    return final_summary, stats_cols, game_cols



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

        # run a filter on the data that weights the latest games more
        # and returns the stats that we want

        # team_data.loc[:, stats] = team_data[stats].rolling(window=filter_window).mean()
        # # chop off the filter window
        # team_data = team_data[filter_window:]


        # def __windowed_average__(x):
        #     # x=x.reset_index(drop=True)
        #     window = pre_compute * discount_weights.loc[x.index].values
        #     return (x.values * window).sum() / window.sum()
        # team_data.loc[:, stats] = team_data[stats].apply(__windowed_average__)
        # team_data.loc[:, stats] = team_data[stats].ewm(alpha=.2).mean()
        #team_data.loc[:, stats] = team_data[stats].rolling(window=filter_window).apply(__windowed_average__)
        # take a windowed average of the last 10 games- argument 0
        # take ewma
        #team_data.loc[:, stats] = team_data[stats].ewm(alpha=alpha).mean()
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

def eval_models(X_train, X_test, y_train, y_test):
    accuracy_scores = []
    for model_name, (model_class, params) in models.items():
        model = model_class(**params)
        # model = HistGradientBoostingClassifier()
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        # conf_matrix = confusion_matrix(y_test, y_pred)
        # class_report = classification_report(y_test, y_pred)

        print("Accuracy:", accuracy)
        # print("Confusion Matrix:\n", conf_matrix)
        # print("Classification Report:\n", class_report)
        accuracy_scores.append((model_name, accuracy))

    accuracy_scores.sort(key=lambda x: x[1], reverse=True)
    print("Accuracy Scores:")
    for model_name, accuracy in accuracy_scores:
        print(f"{model_name}: {accuracy}")

    print()
    # print the dimensions of train data
    print('Number of training games:', X_train.shape[0])
    print('Number of features:', X_train.shape[1])
    print()
    print('Number of test games:', X_test.shape[0])


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

def try_model(X_train, X_test, y_train, y_test, model_name):

    # pca = PCA(n_components=9)
    # X = pca.fit_transform(X)

    
    #model = models[options.model]
    #model = CatBoostClassifier(learning_rate=0.1, iterations=100, depth=6, verbose=0)
    model_class, params = models[model_name]
    model = model_class(**params)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    if not np.isnan(y_test).any():
        accuracy = accuracy_score(y_test, y_pred)
    else:
        accuracy = 0
    #print('Accuracy: %f' % accuracy)
    return y_pred, accuracy


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

    # Fetch the data with caching
    data, stats_cols, game_cols = fetch_data_with_cache(options.start_year, options.end_year, options.end_week, options, current_year)



    alpha = .32
    season_discount = .57
    if options.find_filter_params:
        max_alpha = 0.0
        max_accuracy = 0.0
        for alpha in np.linspace(.01, .6, 20):
            X, Y, vectors = get_records(data, stats_cols, alpha, season_discount, options)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3) # , shuffle=False, shuffle=False
            pred, acc = try_model(X_train, X_test, y_train, y_test, options.model)
            print('Alpha %f - Accuracy %f' % (alpha, acc))
            if acc > max_accuracy:
                max_alpha = alpha
                max_accuracy = acc
        alpha = max_alpha
        max_season_discount = 0.0
        max_accuracy = 0.0
        for season_discount in np.linspace(.05, .95, 25):
            X, Y, vectors = get_records(data, stats_cols, alpha, season_discount, options)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3) # , shuffle=False, shuffle=False
            pred, acc = try_model(X_train, X_test, y_train, y_test, options.model)
            print('Season Discount %f - Accuracy %f' % (season_discount, acc))
            if acc > max_accuracy:
                max_season_discount = season_discount
                max_accuracy = acc
        season_discount = max_season_discount

    print('Alpha: %f' % alpha)
    print('Season discount: %f' % season_discount)

    if options.eval_model:
        # Split the data into training and testing sets
        X, Y, vectors = get_records(data, stats_cols, alpha, season_discount, options)
        # pca = PCA(n_components=6)
        # X = pca.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3) # , shuffle=False, shuffle=False
        print('Traning count: %d' % len(X_train))
        print('Testing count: %d' % len(X_test))
        accuracy_scores = []
        max_accuracy = 0.0
        for model_name, (model_class, params) in models.items():
            pred, accuracy_score = try_model(X_train, X_test, y_train, y_test, model_name)
            accuracy_scores.append((model_name, accuracy_score))


        accuracy_scores.sort(key=lambda x: x[1], reverse=True)
        print("Accuracy Scores:")
        for model_name, accuracy in accuracy_scores:
            print(f"{model_name}: {accuracy}")

    if options.check_season:
        # Split the data into training and testing sets
        X, Y, vectors = get_records(data, stats_cols, alpha, season_discount, options)
        # pca = PCA(n_components=6)
        # X = pca.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3) # , shuffle=False, shuffle=False
        print('Traning count: %d' % len(X_train))
        print('Testing count: %d' % len(X_test))

        pred, accuracy_score = try_model(X_train, X_test, y_train, y_test, options.model)

        # take the predictions and put them back in the 
        eval_df = pd.DataFrame({'actual': y_test, 'pred' : pred})
        vectors = vectors.loc[X_test.index]
        v =vectors.loc[X_test.index]
        eval_df['week'] = v['week_home']
        eval_df['year'] = v['year_home']
        eval_df['correct'] = eval_df['actual'] == eval_df['pred']
        week_val = eval_df.groupby('week').mean()
        print (week_val['correct'])
        year_val = eval_df.groupby('year').mean()
        print(year_val['correct'])




    if options.predict_year and options.predict_week:
        # get the rows that match the year and week
        data = data.copy()
        data_filtered = filter_stats(data, stats_cols, alpha=alpha, season_discount=season_discount)
        if options.season_type:
            data_filtered = data_filtered[data_filtered['season_type'] == options.season_type]

        vectors, new_stats = form_vectors(data_filtered, stats_cols)


        # get the model train rows
        vectors_train = vectors.loc[(vectors['year_home'] < int(options.predict_year)) | ((vectors['year_home'] == int(options.predict_year)) & (vectors['week_home'] < int(options.predict_week)))]
        vectors_train = vectors_train.dropna()

        # get the prediction rows
        vectors_predict = vectors.loc[(vectors['year_home'] == int(options.predict_year)) & (vectors['week_home'] == int(options.predict_week))]


        # Split the data into training and testing sets

        X_train, y_train = get_xy(vectors_train, new_stats)
        X_predict, y_predict = get_xy(vectors_predict, new_stats)

        pred, accuracy_score = try_model(X_train, X_predict, y_train, y_predict, options.model)

        print("Predictions:")
        vectors_predict.reset_index(drop=True, inplace=True)
        for row in vectors_predict.iterrows():
            print(f"Game: {row[1]['game_id']}, Home team: {row[1]['home_team_home']}, Away team: {row[1]['away_team_home']}, Winner: {row[1]['home_team_home'] if int(pred[row[0]]) == 1 else row[1]['away_team_home']}")



if __name__ == "__main__":
    main()



    # print_correlation(data, 'drive_score_percentage')
    # print_correlation(data, 'drive_score_percentage_def')
    # print_correlation(data, 'score')
    # print_correlation(data, 'def_score')
    # sys.exit(0)


