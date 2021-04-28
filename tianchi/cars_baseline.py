import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMRegressor
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 设置与一下表格大小 防止出现省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

## 通过Pandas对于数据进行读取 (pandas是一个很友好的数据读取函数库)
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(train.head(10))
print(train.describe())
print(train.isnull().any())
print('-----------------------------------------------分割线--------------------------------------------------------')
print(train[['model', 'bodyType', 'fuelType', 'gearbox']].describe())  # 存在缺失值
print(train.dropna().describe())  # 删除缺失值
train.fillna(-999, inplace=True)
test.fillna(-999, inplace=True)
print(train.columns)
price = train['price']
price_log = np.log(price)

plt.figure(figsize=(20, 15))
plt.subplot(4, 5, 1)
sns.distplot(price)
plt.subplot(4, 5, 2)
sns.distplot(price_log, axlabel='price_log')
plt.subplot(4, 5, 3)
price_log.plot.box()
plt.subplot(4, 5, 4)
sns.distplot(train['name'], axlabel='train_name')
plt.subplot(4, 5, 5)
sns.distplot(test['name'], axlabel='test_name')
plt.subplot(4, 5, 6)
sns.distplot(np.log(test['name'] + 0.00001), axlabel='test_name_log')
plt.subplot(4, 5, 7)
sns.distplot(train['model'], axlabel='train_model')
plt.subplot(4, 5, 8)
sns.distplot(test['model'], axlabel='test_model')
plt.subplot(4, 5, 9)
sns.distplot(np.log(test['model'] + 0.0001), axlabel='test_model_log')
plt.subplot(4, 5, 10)
sns.distplot(train['brand'], axlabel='train_brand')
plt.subplot(4, 5, 11)
sns.distplot(test['brand'], axlabel='test_brand')
plt.subplot(4, 5, 12)
sns.distplot(train['bodyType'], axlabel='train_bodyType')
plt.subplot(4, 5, 13)
sns.distplot(test['bodyType'], axlabel='test_bodyType')
plt.subplot(4, 5, 14)
sns.distplot(train['fuelType'], axlabel='train_fuelType')
plt.subplot(4, 5, 15)
sns.distplot(test['fuelType'], axlabel='test_fuleType')
plt.subplot(4, 5, 16)
sns.distplot(train['gearbox'], axlabel='train_gearbox')
plt.subplot(4, 5, 17)
sns.distplot(test['gearbox'], axlabel='test_gearbox')
plt.subplot(4, 5, 18)
sns.distplot(train['power'], axlabel='train_power')
plt.subplot(4, 5, 19)
sns.distplot(test['power'], axlabel='test_power')
# plt.show()
print('-----------------------------------------------分割线--------------------------------------------------------')

# 根据题目，大于600的属于非正常范围，考虑 train上删除记录，test上暂定截断
print(train['power'].describe())

train = train.drop('price', axis=1)
train['notRepairedDamage'] = train['notRepairedDamage'].map(lambda x: -1 if (x == '-') else x)
train['notRepairedDamage'] = train['notRepairedDamage'].map(lambda x: float(x))
test['notRepairedDamage'] = test['notRepairedDamage'].map(lambda x: -1 if (x == '-') else x)
test['notRepairedDamage'] = test['notRepairedDamage'].map(lambda x: float(x))

plt.figure(figsize=(20, 15))
plt.subplot(4, 5, 1)
sns.distplot(train['kilometer'], axlabel='train_kilometer')
plt.subplot(4, 5, 2)
sns.distplot(test['kilometer'], axlabel='test_kilometer')
plt.subplot(4, 5, 3)
sns.distplot(train['notRepairedDamage'], axlabel='train_notRepairedDamage')
plt.subplot(4, 5, 4)
sns.distplot(test['notRepairedDamage'], axlabel='test_notRepairedDamage')
plt.subplot(4, 5, 5)
sns.distplot(train['regionCode'], axlabel='train_regionCode')
plt.subplot(4, 5, 6)
sns.distplot(test['regionCode'], axlabel='test_regionCode')
plt.subplot(4, 5, 7)
sns.distplot(train['seller'], axlabel='train_seller')
plt.subplot(4, 5, 8)
sns.distplot(test['seller'], axlabel='test_seller')
plt.subplot(4, 5, 9)
sns.distplot(train['offerType'], axlabel='train_offerType')
plt.subplot(4, 5, 10)
sns.distplot(test['offerType'], axlabel='test_offerType')
# plt.show()
print('-----------------------------------------------分割线--------------------------------------------------------')
# seller、offerType字段 test上就一个值 木什么用 可以拿掉

plt.figure(figsize=(20, 15))
plt.subplot(4, 5, 1)
sns.distplot(train['creatDate'].map(lambda x: int(str(x)[:4])), axlabel='train_creat_year')
plt.subplot(4, 5, 2)
sns.distplot(train['creatDate'].map(lambda x: int(str(x)[:4])), axlabel='test_creat_year')
plt.subplot(4, 5, 3)
sns.distplot(train['creatDate'].map(lambda x: int(str(x)[4:6])), axlabel='train_creat_month')
plt.subplot(4, 5, 4)
sns.distplot(test['creatDate'].map(lambda x: int(str(x)[4:6])), axlabel='test_creat_month')
plt.subplot(4, 5, 5)
sns.distplot(train['creatDate'].map(lambda x: int(str(x)[6:])), axlabel='train_create_day')
plt.subplot(4, 5, 6)
sns.distplot(test['creatDate'].map(lambda x: int(str(x)[6:])), axlabel='test_creat_day')
plt.subplot(4, 5, 7)
sns.distplot(train['v_1'], axlabel='train_v1')
plt.subplot(4, 5, 8)
sns.distplot(test['v_1'], axlabel='test_v1')
plt.subplot(4, 5, 9)
sns.distplot(train['v_2'], axlabel='train_v2')
plt.subplot(4, 5, 10)
sns.distplot(test['v_2'], axlabel='test_v2')
plt.subplot(4, 5, 11)
sns.distplot(train['v_3'], axlabel='train_v3')
plt.subplot(4, 5, 12)
sns.distplot(test['v_3'], axlabel='test_v3')
plt.subplot(4, 5, 13)
sns.distplot(train['v_4'], axlabel='train_v4')
plt.subplot(4, 5, 14)
sns.distplot(test['v_4'], axlabel='test_v4')
plt.subplot(4, 5, 15)
sns.distplot(train['v_5'], axlabel='train_v5')
plt.subplot(4, 5, 16)
sns.distplot(test['v_5'], axlabel='test_v5')
plt.subplot(4, 5, 17)
sns.distplot(train['v_6'], axlabel='train_v6')
plt.subplot(4, 5, 18)
sns.distplot(test['v_6'], axlabel='test_v6')
plt.subplot(4, 5, 19)
sns.distplot(train['v_7'], axlabel='train_v7')
plt.subplot(4, 5, 20)
sns.distplot(test['v_7'], axlabel='test_v7')
# plt.show()
print('-----------------------------------------------分割线--------------------------------------------------------')

plt.figure(figsize=(20, 15))
plt.subplot(4, 5, 1)
sns.distplot(train['v_8'], axlabel='train_v8')
plt.subplot(4, 5, 2)
sns.distplot(test['v_8'], axlabel='test_v8')
plt.subplot(4, 5, 3)
sns.distplot(train['v_9'], axlabel='train_v9')
plt.subplot(4, 5, 4)
sns.distplot(test['v_9'], axlabel='test_v9')
plt.subplot(4, 5, 5)
sns.distplot(train['v_10'], axlabel='train_v10')
plt.subplot(4, 5, 6)
sns.distplot(test['v_10'], axlabel='test_v10')
plt.subplot(4, 5, 7)
sns.distplot(train['v_11'], axlabel='train_v11')
plt.subplot(4, 5, 8)
sns.distplot(test['v_11'], axlabel='test_v11')
plt.subplot(4, 5, 9)
sns.distplot(train['v_12'], axlabel='train_v12')
plt.subplot(4, 5, 10)
sns.distplot(test['v_12'], axlabel='test_v12')
plt.subplot(4, 5, 11)
sns.distplot(train['v_13'], axlabel='train_v13')
plt.subplot(4, 5, 12)
sns.distplot(test['v_13'], axlabel='test_v13')
plt.subplot(4, 5, 13)
sns.distplot(train['v_14'], axlabel='train_v14')
plt.subplot(4, 5, 14)
sns.distplot(test['v_14'], axlabel='test_v14')
# plt.show()
print('-----------------------------------------------分割线--------------------------------------------------------')
# 观察到训练集上测试集上v13 v14分布有些不一致
print(train[['v_13', 'v_14']].describe())

print(test[['v_13', 'v_14']].describe())
print(train[['regDate', 'creatDate']].head(10))

train['year_var'] = train['creatDate'].map(lambda x: int(str(x)[:4])) - train['regDate'].map(lambda x: int(str(x)[:4]))
train['month_var'] = train['creatDate'].map(lambda x: int(str(x)[4:6])) - train['regDate'].map(lambda x: int(str(x)[4:6]))
train['day_var'] = train['creatDate'].map(lambda x: int(str(x)[6:])) - train['regDate'].map(lambda x: int(str(x)[6:]))
test['year_var'] = test['creatDate'].map(lambda x: int(str(x)[:4])) - test['regDate'].map(lambda x: int(str(x)[:4]))
test['month_var'] = test['creatDate'].map(lambda x: int(str(x)[4:6])) - test['regDate'].map(lambda x: int(str(x)[4:6]))
test['day_var'] = test['creatDate'].map(lambda x: int(str(x)[6:])) - test['regDate'].map(lambda x: int(str(x)[6:]))
print('-----------------------------------------------分割线--------------------------------------------------------')

plt.figure(figsize=(10, 10))
plt.subplot(2, 3, 1)
sns.distplot(train['year_var'], axlabel='train_year_var')
plt.subplot(2, 3, 2)
sns.distplot(train['month_var'], axlabel='train_month_var')
plt.subplot(2, 3, 3)
sns.distplot(train['day_var'], axlabel='train_day_var')
plt.subplot(2, 3, 4)
sns.distplot(test['year_var'], axlabel='test_year_var')
plt.subplot(2, 3, 5)
sns.distplot(test['month_var'], axlabel='test_month_var')
plt.subplot(2, 3, 6)
sns.distplot(test['day_var'], axlabel='test_day_var')
# plt.show()
print('-----------------------------------------------分割线--------------------------------------------------------')

print(len(train.columns))
print(len(test.columns))
print(train.shape)

mask = (train['power'] < 600) & (train['seller'] == 0) & (train['v_13'] < 6) & (train['v_14'] < 2.8)
train = train[mask]
price_log = price_log[mask]
price = price[mask]
print(train.shape)

train.drop(['seller', 'offerType'], axis=1, inplace=True)
test.drop(['seller', 'offerType'], axis=1, inplace=True)
test['power'] = test['power'].map(lambda x: x if (x < 600) else 600)  # 将小于600的值设为600
corr = train.corr()  # 相关系数矩阵,即给出任意两列之间的相关系数
sns.heatmap(corr, )
# plt.show()

print(train.columns)
print(train.head())
print(test.columns)
print(test.head())

print('-----------------------------------------------分割线--------------------------------------------------------')

train_x, val_x, train_y, val_y = train_test_split(train, price_log)
lgb1 = LGBMRegressor(n_estimators=2000, num_leaves=90, max_depth=13, early_stopping_round=50, metric=['l1'])
lgb1.fit(train_x, train_y, eval_set=[(val_x, val_y)], eval_metric='l1')
print(lgb1.feature_importances_)
print('-----------------------------------------------分割线----------------------------------------------------')
train.columns[lgb1.feature_importances_ > 5000]


def col_mul(data):
    cols1 = ['SaleID', 'name', 'regDate', 'power', 'regionCode', 'creatDate', 'v_0',
             'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10',
             'v_11', 'v_12', 'v_13', 'v_14', 'day_var']
    cols2 = ['SaleID', 'name', 'regDate', 'power', 'regionCode', 'creatDate', 'v_0',
             'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10',
             'v_11', 'v_12', 'v_13', 'v_14', 'day_var']
    result = pd.DataFrame()
    i = 1
    for col1 in cols1:
        if (len(cols2[i:]) == 0):
            pass
        for col2 in cols2[i:]:
            if (col2 == col1):
                pass
            new_col = data[col1] * data[col2]
            result[col1 + '_' + 'mul' + '_' + col2] = new_col
            ++i
    return result


train_ = pd.concat([train, col_mul(train)], axis=1)
train_x1, val_x1, train_y1, val_y1 = train_test_split(train_, price_log)
lgb2 = LGBMRegressor(n_estimators=2000, num_leaves=90, max_depth=13, early_stopping_round=50, metric=['l1'])
lgb2.fit(train_x1, train_y1, eval_set=[(val_x1, val_y1)], eval_metric='l1')
print(lgb2.feature_importances_)

print('-----------------------------------------------分割线----------------------------------------------------')
lgb3 = LGBMRegressor(n_estimators=1970, num_leaves=90, max_depth=13)
lgb3.fit(train, price_log)

result = lgb3.predict(test)
result = np.exp(result)
result = pd.Series(result)
result = pd.concat([test['SaleID'], result], axis=1)
result.columns = ['SaleID', 'price']
print(result.head(10))
result.to_csv('result2.csv', index=None)
