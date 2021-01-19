import numpy as np
import pandas as pd
from sklearn import preprocessing
"""
changed from https://www.kaggle.com/anycode/simple-nn-baseline

to run this kernel, pip install ultimate first from your custom packages
"""
import gc, sys
import lightgbm as lgb
import pickle
gc.enable()
pd.set_option('display.max_columns', 500)


def feature_engineering(is_train=True, debug=True, type='squad-fpp', flag = True):
    test_idx = None
    if is_train:
        print("processing train.csv")
        if debug == True:
            df = pd.read_csv('C:\\Users\\lenovo\\Desktop\\term1\\大数据解析\\小组PJ\\pubg-finish-placement-prediction\\train_V2.csv')
        else:
            df = pd.read_csv('C:\\Users\\lenovo\\Desktop\\term1\\大数据解析\\小组PJ\\pubg-finish-placement-prediction\\train_V2.csv')

        df = df[df['maxPlace'] > 1]

        winPlacePerc = df['winPlacePerc'].values
        maxPlace = df['maxPlace'].values
        numList = df['numGroups'].values
        rankList = []
        for i in range(len(winPlacePerc)):
            cur = winPlacePerc[i] * (maxPlace[i] - 1) + 1
            curn = numList[i]
            newwinPlacePerc = (cur - 1) / (curn - 1)
            rankList.append(newwinPlacePerc)
        rankArr = np.array(rankList)
        df['winPlacePerc'] = rankArr
        print('rankArr:')
        print(rankArr)
    else:
        print("processing test.csv")
        df = pd.read_csv('C:\\Users\\lenovo\\Desktop\\term1\\大数据解析\\小组PJ\\pubg-finish-placement-prediction\\test_V2.csv')
        test_idx = df['Id'].values

    print(df.columns)
    special = ['crashfpp', 'crashtpp', 'flarefpp', 'flaretpp', 'normal-duo',
        'normal-duo-fpp', 'normal-solo', 'normal-solo-fpp', 'normal-squad', 'normal-squad-fpp']
    if flag:
        df = df[df['matchType']==type]
    else:
        df = df[df['matchType'].isin(special)]
    print('len(df):')
    print(len(df))
    print("remove some columns")
    target = 'winPlacePerc'

    print("Adding Features")

    df['headshotrate'] = df['kills'] / df['headshotKills']
    df['killStreakrate'] = df['killStreaks'] / df['kills']
    df['healthitems'] = df['heals'] + df['boosts']
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']
    df['distance_over_weapons'] = df['totalDistance'] / df['weaponsAcquired']
    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']
    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']
    df['killsPerWalkDistance'] = df['kills'] / df['walkDistance']
    df["skill"] = df["headshotKills"] + df["roadKills"]

    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN

    print("Removing Na's From DF")
    df.fillna(0, inplace=True)

    features = list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchType")

    y = None

    if is_train:
        print("get target")
        y = np.array(df.groupby(['matchId', 'groupId'])[target].agg('mean'), dtype=np.float64)
        features.remove(target)

    print("get group mean feature")
    agg = df.groupby(['matchId', 'groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    if is_train:
        df_out = agg.reset_index()[['matchId', 'groupId']]
    else:
        df_out = df[['matchId', 'groupId']]

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])

    print("get group max feature")
    agg = df.groupby(['matchId', 'groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])

    print("get group min feature")
    agg = df.groupby(['matchId', 'groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])

    print("get group size feature")
    agg = df.groupby(['matchId', 'groupId']).size().reset_index(name='group_size')
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])

    print("get match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])

    # print("get match type feature")
    # agg = df.groupby(['matchId'])[matchType.columns].agg('mean').reset_index()
    # df_out = df_out.merge(agg, suffixes=["", "_match_type"], how='left', on=['matchId'])

    print("get match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    df_out = df_out.merge(agg, how='left', on=['matchId'])

    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)

    X = df_out

    feature_names = list(df_out.columns)

    del df, df_out, agg, agg_rank
    gc.collect()
    print('len(X):')
    print(len(X))
    if not is_train:
        print('len(test_idx):')
        print(len(test_idx))
    return X, y, feature_names, test_idx


# 这里看明白训练集和测试集是同一个形式的数据集。
# custom function to run light gbm model


def run_lgb(train_X, train_y, val_X, val_y, x_test):
    params = {"objective": "regression", "metric": "mae", 'n_estimators': 20000, 'early_stopping_rounds': 200,
              "num_leaves": 31, "learning_rate": 0.05, "bagging_fraction": 0.7,
              "bagging_seed": 0, "num_threads": 4, "colsample_bytree": 0.7
              }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, valid_sets=[lgtrain, lgval], early_stopping_rounds=200, verbose_eval=1000)

    # pred_test_y = model.predict(x_test, num_iteration=model.best_iteration)
    return model


def train_models(Traintype):
    if Traintype != 'combine':
        x_train, y_train, train_columns, _ = feature_engineering(True, True, Traintype, True)
        x_test, _, _ , test_idx = feature_engineering(False, True, Traintype, True)
    else:
        x_train, y_train, train_columns, _ = feature_engineering(True, True, Traintype, False)
        x_test, _, _, test_idx = feature_engineering(False, True, Traintype, False)
    train_index = round(int(x_train.shape[0]*0.8))
    dev_X = x_train[:train_index]
    val_X = x_train[train_index:]
    dev_y = y_train[:train_index]
    val_y = y_train[train_index:]
    gc.collect()
    model = run_lgb(dev_X, dev_y, val_X, val_y, x_test)
    # 记录这里的pred_test。
    file = open('LGBM_3_'+Traintype+'(1st).pickle', 'wb')
    pickle.dump(model, file)
    return test_idx


typeList = ['solo', 'solo-fpp', 'duo', 'duo-fpp', 'squad', 'squad-fpp', 'combine']
indexes = []
prediction = []
for eachtype in typeList:
    index = train_models(eachtype)
    indexes.extend(index)