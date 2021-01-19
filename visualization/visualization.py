import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.ticker as mtick
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")
train = pd.read_csv('C:\\Users\\lenovo\\Desktop\\term1\\大数据解析\\小组PJ\\pubg-finish-placement-prediction\\train_V2.csv') #加载数据


def kills(ax):
    data = train.copy()
    data.loc[data['kills'] > data['kills'].quantile(0.99)] = 8
    data.sort_values(['kills'], ascending=True)
    gp_col = 'kills'
    kills_count = data.groupby(gp_col).size()
    b = []
    for i in kills_count:
        b.append(i)
    c = ['0', '1', '2', '3', '4', '5', '6', '7', '7+']
    color = ['#FDE255', '#FDE255', '#FDE255', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray']
    # plt.figure(figsize=(12, 8))
    ax.bar(c, b, color=color)
    plt.ylabel('frequency')
    plt.xlabel('Kills')
    plt.xticks()
    plt.yticks()
    # plt.legend()
    plt.title('Kills distribution')
    del data
    # plt.text("击杀数分布直方图", fontsize=15)
    # plt.show()  # 80%的玩家击杀数都小于等于2，即黄色部分


def assists(ax):
    # 玩家助攻数分布直方图
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    data = train.copy()
    data.loc[data['assists'] > data['assists'].quantile(0.99)] = 4
    data.sort_values(['assists'], ascending=True)
    gp_col = 'assists'
    kills_count = data.groupby(gp_col).size()
    b = []
    for i in kills_count:
        b.append(i)
    c = ['0', '1', '2', '3', '4+']
    color = ['#FDE255', '#FDE255', 'gray', 'gray', 'gray']
    # ax.subplot()
    ax.bar(c, b, color=color)
    plt.ylabel('frequency')
    plt.xlabel('Assists')
    plt.xticks()
    plt.yticks()
    plt.title("Assists distribution")
    del data
    # plt.show()  # 80%的玩家助攻数都小于等于1，即黄色部分


def boosts(ax):
    # 玩家加速剂使用数分布直方图
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    data = train.copy()
    data.loc[data['boosts'] > data['boosts'].quantile(0.99)] = 8
    data.sort_values(['boosts'], ascending=True)
    gp_col = 'boosts'
    kills_count = data.groupby(gp_col).size()
    b = []
    for i in kills_count:
        b.append(i)
    c = ['0', '1', '2', '3', '4', '5', '6', '7', '7+']
    color = ['#FDE255', '#FDE255', '#FDE255', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray']
    # plt.figure(figsize=(12, 8))
    ax.bar(c, b, color=color)
    # plt.ylabel('频数', fontsize=15)
    # plt.xlabel('加速剂使用数', fontsize=15)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    plt.ylabel('frequency')
    plt.xlabel('Boosts')
    plt.xticks()
    plt.yticks()
    plt.title("Boosts distribution")
    del data
    # plt.show()  # 75%的玩家击杀数都小于等于2，即黄色部分


def vehicleDestroys(ax):
    data = train.copy()
    data.sort_values(['assists'], ascending=True)
    gp_col = 'vehicleDestroys'
    kills_count = data.groupby(gp_col).size()
    b = []
    for i in kills_count:
        b.append(i)
    c = ['0', '1', '2', '3', '4', '5']
    color = ['#FDE255', 'gray', 'gray', 'gray', 'gray', 'gray']
    # plt.figure(figsize=(12, 8))
    ax.bar(c, b, color=color)
    plt.ylabel('frequency')
    plt.xlabel('vehicleDestroys')
    plt.xticks()
    plt.yticks()
    plt.title("vehicleDestroys distribution")
    del data
    # plt.show()  # 99.5%的玩家助攻数都小于等于1，即黄色部分


def weaponsAcquired(ax):
    data = train.copy()
    data.loc[data['weaponsAcquired'] > data['weaponsAcquired'].quantile(0.99)] = 11
    data.sort_values(['weaponsAcquired'], ascending=True)
    gp_col = 'weaponsAcquired'
    kills_count = data.groupby(gp_col).size()
    b = []
    for i in kills_count:
        b.append(i)
    c = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '10+']
    color = ['gray', '#FDE255', '#FDE255', '#FDE255', '#FDE255', '#FDE255', '#FDE255', 'gray', 'gray', 'gray', 'gray',
             'gray']
    # plt.figure(figsize=(12, 8))
    ax.bar(c, b, color=color)
    plt.ylabel('frequency')
    plt.xlabel('Weapons Acquired')
    plt.xticks()
    plt.yticks()
    plt.axis()
    plt.title("Weapons Acquired distribution")
    del data
    # plt.show()  # 80%的玩家获得的武器数量都介于1到6，即黄色部分


def heals(ax):
    data = train.copy()
    data.loc[data['revives'] > data['revives'].quantile(0.995)] = 3
    data.sort_values(['revives'], ascending=True)
    gp_col = 'revives'
    kills_count = data.groupby(gp_col).size()
    b = []
    for i in kills_count:
        b.append(i)
    c = ['0', '1', '2', '2+']
    color = ['#FDE255', '#FDE255', '#FDE255', 'gray']
    # plt.figure(figsize=(12, 8))
    ax.bar(c, b, color=color)
    plt.ylabel('frequency')
    plt.xlabel('Revives')
    plt.xticks()
    plt.yticks()
    plt.title("Revives distribution")
    del data
    # plt.show()  # 99.5%的玩家复活队友数都小于等于2，即黄色部分


def rankPoints():
    # data = train.copy()
    data = train[train['rankPoints'] < train['rankPoints'].quantile(0.99)]
    # plt.subplot(311)
    plt.title("rankPoints distribution")
    sns.distplot(data['rankPoints'], kde_kws={"color": "gray"}, hist_kws={"alpha": 1, "color": "#FDE255"})
    plt.ylabel('proportion')
    plt.xlabel('rankPoints')
    plt.xticks()
    plt.yticks()
    del data


def winPoints():
    data = train.copy()
    data = data[data['winPoints'] < data['winPoints'].quantile(0.99)]
    # plt.subplot(312)
    plt.title("win")
    sns.distplot(data['winPoints'], kde_kws={"color": "gray"}, hist_kws={"alpha": 1, "color": "#FDE255"})
    plt.ylabel('比例')
    plt.xlabel('获胜得分')
    plt.xticks()
    plt.yticks()
    del data


def killPoints():
    data = train.copy()
    data = data[data['killPoints'] < data['killPoints'].quantile(0.99)]
    # plt.subplot(313)
    plt.title("获胜得分分布图")
    sns.distplot(data['winPoints'], kde_kws={"color": "gray"}, hist_kws={"alpha": 1, "color": "#FDE255"})
    plt.ylabel('比例')
    plt.xlabel('获胜得分')
    plt.xticks()
    plt.yticks()
    del data


ax = plt.subplot(231)
kills(ax)
ax1 = plt.subplot(232)
assists(ax1)
ax2 = plt.subplot(233)
boosts(ax2)
ax3 = plt.subplot(234)
vehicleDestroys(ax3)
ax4 = plt.subplot(235)
weaponsAcquired(ax4)
ax5 = plt.subplot(236)
heals(ax5)
plt.tight_layout()
plt.show()
# rankPoints()
# plt.tight_layout()
# plt.show()
# winPoints()
# plt.tight_layout()
# plt.show()
# killPoints()
# plt.tight_layout()
# plt.show()
