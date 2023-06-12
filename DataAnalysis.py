import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from pathlib import Path as P

indir = P("./icr-identify-age-related-conditions")
train = pd.read_csv(indir.joinpath("train.csv"))
test = pd.read_csv(indir.joinpath("test.csv"))
greeks = pd.read_csv(indir.joinpath("greeks.csv"))
sample_submission = pd.read_csv(indir.joinpath("sample_submission.csv"))

# print(train.shape)
# print(train.info())

# 查缺失值
missing_counts = train.isnull().sum()
# print(missing_counts)
# print(train.describe().T)


# 检查是否有重复数据
def check_duplicate(df):
    if df.duplicated().all():
        return "There are dulicate data"
    else:
        return "data is clean"


# print(check_duplicate(train))


# 可视化缺失值
all_cols = train.columns
# sns.heatmap(train[all_cols].isnull(),cmap='viridis')
# plt.show()

# 使用均值填充缺失值,有些冗余可以封装一下      ^_^
train["EL"] = train["EL"].fillna((train["EL"].median()))
train["BQ"] = train["BQ"].fillna((train["BQ"].median()))
train["CC"] = train["CC"].fillna((train["CC"].median()))
train["FS"] = train["FS"].fillna((train["FS"].median()))
train["CB"] = train["CB"].fillna((train["CB"].median()))
train["GL"] = train["GL"].fillna((train["GL"].median()))
train["DU"] = train["DU"].fillna((train["DU"].median()))
train["FL"] = train["FL"].fillna((train["FL"].median()))
train["FC"] = train["FC"].fillna((train["FC"].median()))

# sns.heatmap(train[all_cols].isnull(),cmap='viridis')
# plt.show()


# 样本标签是否均衡？
def label_plotting(df, col, title):
    """
    :param df:数据
    :param col: 列名
    :param title: 图标
    :return: 无
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.countplot(
        x=col, data=df, palette="flare", order=df[col].value_counts().index
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.bar_label(ax.containers[0])
    plt.title(title)
    plt.show()


# label_plotting(train,'Class',"label counting")
# 508个没有被诊断，108个已被诊断为3中情况之一
# 需要oversamping解决

# 由于列 EJ代表的是 categorical 需要将其转换为数字
train["EJ"] = train["EJ"].replace({"A": 0, "B": 1})
test["EJ"] = test["EJ"].replace({"A": 0, "B": 1})

# 筛选出输入特征向量X与labelY
X = train.drop(["Id", "Class"], axis=True)
Y = train["Class"]

# 数据归一化处理 消除实验差异
StandardS = StandardScaler()
normalization_X = StandardS.fit_transform(X)

# 过采样用imblearn库也可自己写
# https://imbalanced-learn.org/stable/over_sampling.html api官网
oversample = RandomOverSampler(random_state=0)
temp_X, temp_Y = oversample.fit_resample(normalization_X, Y)

counter = Counter(temp_Y)
for k, v in counter.items():
    per = v / len(temp_Y) * 100
    print("Class=%d,n=%d (%.3f%%)" % (k, v, per))
    
plt.bar(counter.keys(), counter.values())
plt.show()

x_train, x_test, y_train, y_test = train_test_split(
    temp_X, temp_Y, test_size=0.3, random_state=30, shuffle=True
)
# base line 分类
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print("Precision value:{:.2f} %".format(precision_score(y_test, y_pred)))
print("Recall value:{:.2f} %".format(recall_score(y_test, y_pred)))
print("Accuracy value:{:.2f} %".format(accuracy_score(y_test, y_pred)))
