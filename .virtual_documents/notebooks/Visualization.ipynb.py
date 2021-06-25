import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib as mplt
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic("matplotlib", " inline")


TRAIN_PATH = "../data/train.csv"
TEST_PATH = '../data/test.csv'
Full_train = pd.read_csv(TRAIN_PATH,index_col='Id')
Full_test = pd.read_csv(TEST_PATH)
Full_train.dropna(axis=0,subset=['SalePrice'], inplace=True)
y_full = Full_train.SalePrice
X_full = Full_train.drop(axis=1, columns=['SalePrice'])


Full_train = Full_train.rename(columns={'SalePrice' : 'price'})
Full_train


columns_to_drop = ['MoSold','YrSold'] # these columns could be cause of data leakage, it is better to drop them
X_full.drop(columns_to_drop, axis=1, inplace=True)


numeric_col = list((X_full.select_dtypes(exclude='object')).columns)
string_col = list((X_full.select_dtypes(include='object')).columns)
low_categorical_col = [col for col in X_full.columns
                  if X_full[col].dtype == 'object' and X_full[col].nunique() <10]


# to see dtypes of all columns, we need change max_rows
pd.set_option('display.max_rows', 80)
Full_train.dtypes


Full_train.isna().sum()[Full_train.isna().sum() > 0]


Full_train[numeric_col].isna().sum()[Full_train[numeric_col].isna().sum() > 0]


missed_numeric_values = Full_train[numeric_col].isna().sum()[Full_train[numeric_col].isna().sum() > 0].index
print(missed_numeric_values)


full_x_corr = Full_train.corr()


plt.figure(figsize=(14,9))
sns.heatmap(full_x_corr, mask=full_x_corr < 0.5, xticklabels=True, yticklabels=True)


full_x_corr.loc[full_x_corr['price'] > 0.5, 'price']


good_correlation = full_x_corr.loc[full_x_corr['price'] > 0.5, 'price'].index
good_correlation


from_num_to_cat = ['GarageCars', 'GarageYrBlt', 'YearBuilt', 'BedroomAbvGr', '2ndFlrSF']


feature = 'GarageYrBlt'
bins = np.linspace(Full_train[feature].min(), Full_train[feature].max(), 20)
years = pd.cut(Full_train[feature], bins=bins, include_lowest=True)


le = LabelEncoder()
le.fit(years)
encoded = pd.Series(le.transform(years))


y=Full_train['price'].reset_index().price


encoded.to_frame().join(y).corr()


encoded.to_frame().join(Full_train['GarageArea'].reset_index()['GarageArea']).corr()


year = Full_train[feature].copy()
year


corr = pd.get_dummies(years, dummy_na=True).join(Full_train['GarageArea']).corr()
corr['GarageArea'][abs(corr['GarageArea']) > 0.5]


year = year.isna()


year.to_frame().join(Full_train['GarageArea']).corr()


feature = 'YearBuilt'
bins = np.linspace(Full_train[feature].min(), Full_train[feature].max(), 13)
years = pd.cut(Full_train[feature], bins=bins, include_lowest=True)


from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
le.fit(years)


encoded = le.transform(years)


pd.Series(encoded).to_frame().join(y).corr()


corr = pd.get_dummies(years, dummy_na=True).join(Full_train['price']).corr()
corr['price']


feature  = 'BedroomAbvGr'
Full_train[feature]


feature = '2ndFlrSF'
Full_train[feature]


le = LabelEncoder()
bb = pd.cut(Full_train[feature], bins=[0,0.01,1000,2065], include_lowest=True)
le.fit(bb)
pd.Series(le.transform(bb)).to_frame().join(Full_train['price']).corr()


missed_num_df = Full_train[missed_numeric_values].copy()


missed_num_df.isna().sum()


missed_num_df.drop(columns='GarageYrBlt', inplace=True)


missed_num_df.corrwith(Full_train['price'])


missed_num_df


missed_num_df['MasVnrArea'].value_counts()


861/1460


missed_num_df.drop(columns='MasVnrArea', inplace=True)





F_P = {}
for feature in numeric_col:
    y = pd.Series(Full_train['price'], index=Full_train[feature].dropna().index)
    pearson_coef, p_value = stats.pearsonr(Full_train[feature].dropna(),y)
    #if (abs(pearson_coef) > 0.5):
    F_P[feature] = (pearson_coef, p_value)
    #print('Feature:',feature)
    #print('The correlation coefficient is:', pearson_coef, 'the p-value is:', p_value)
for k in F_P:
    print(k, F_P[k])


for k in F_P:
    if F_P[k][0] > 0.5:
        print(k, F_P[k])


for k in F_P:
    if F_P[k][1] < 0.05:
        print(k, F_P[k])


porch = ['EnclosedPorch', 'OpenPorchSF','3SsnPorch','ScreenPorch']
BSM_bath = ['BsmtFullBath', 'BsmtHalfBath']
bath = ['FullBath', 'HalfBath']


for col in Full_train.columns:
    if 'area' in str.lower(col):
        print(col)


areas = ['LotArea', 'MasVnrArea', 'GrLivArea','GarageArea','PoolArea']


selected_features_df = Full_train[price_corr.index].copy(deep=True)
selected_features_df['Has_pool'] = Full_train['PoolArea'] > 0
selected_features_df['Has_fireplace'] = Full_train['Fireplaces'] > 0
selected_features_df['Total_porch'] = Full_train[porch].sum(axis=1)
selected_features_df['Total_BSM_bath'] = Full_train[BSM_bath].sum(axis=1)
selected_features_df['Has_2flr'] = Full_train['2ndFlrSF'] > 0
selected_features_df['Total_Bath'] = Full_train[bath].sum(axis=1)
selected_features_df['Has_garage'] = Full_train['GarageArea'] > 0
selected_features_df['Total_area'] = Full_train[areas].sum(axis=1)
selected_features_df


bools = ['Has_pool', 'Has_fireplace', 'Has_2flr', 'Has_garage']


selected_features_df[bools] = selected_features_df[bools].astype(int)
selected_features_df.dtypes





new_added_cols = [col for col in selected_features_df.columns
                 if col not in Full_train.columns]
print(new_added_cols)


for feature in new_added_cols:
    pearson_coef, p_value = stats.pearsonr(selected_features_df[feature],y_full)
    #if (abs(pearson_coef) > 0.5):
    print('Feature:',feature)
    print('The correlation coefficient is:', pearson_coef, 'the p-value is:', p_value)


sns.boxplot(x='Total_Bath', y='SalePrice', data=selected_features_df)


sns.regplot(x='Total_Bath', y='SalePrice', data=selected_features_df, order=3)





features_num_good_corr = list(price_corr.sort_values(ascending=False).index)
features_num_good_corr += ['Total_Bath']
features_num_good_corr = features_num_good_corr[1:] # remove SalePrice
print(features_num_good_corr[1:])


price_corr.sort_values(ascending=False)


low_cardinality_df = Full_train[low_categorical_col]
low_cardinality_df


low_cardinality_df.isna().sum()[low_cardinality_df.isna().sum() > 0]


low_cardinality_cols_with_nans = list(low_cardinality_df.isna().sum()[low_cardinality_df.isna().sum() > 0].index)


without_nan_df = low_cardinality_df.dropna(axis=1)
without_nan_df


without_nan_df.describe(include=['object'])


without_nan_df = without_nan_df.join(y_full)


def get_dic_name_need_change():
    dic = {}
    y = find_n_percent_of_data(Full_train.shape[0], 5)
    for col_info in cols_with_big_F_score+cols_with_low_F_score:
        col_name = col_info[0]
        values_less_then_y = Full_train[col_name].value_counts() < y    
        values = []
        for i in range(1,len(values_less_then_y)):
            group_value_counts = values_less_then_y.iloc[-i]
            if group_value_counts:
                group_value = values_less_then_y.index[-i]
                values.append(group_value)

        if len(values) > 0:
            dic[values_less_then_y.name] = values
    return dic


def score_ANOVA(data):
    '''
    score ANOVA for all columns in data. Last column must be SalePrice - this is the exit point.
    return 3 lists:
    cols_that_need_further_investigation - ANOVA <=10;
    cols_with_big_F_score - ANOVA > 50
    cols_with_low_F_score - ANOVA <= 50
    '''
    cols_that_need_further_investigation = []
    cols_with_big_F_score = []
    cols_with_low_F_score = []
    
    for col in data.columns:
        if col == 'SalePrice':
            break
        df = data[[col,'SalePrice']]
        grouped_col_price = df.groupby([col],as_index=False)
        uniques_col_values = df[col].unique()
        if len(uniques_col_values) == 1:
            print(f'cols has only one value! ({col})')
            continue
        args = [grouped_col_price.get_group(uniques_col_values[i])['SalePrice'] for i in range(len(uniques_col_values))]
        f_val, p_val = stats.f_oneway(*args)
        #print('column name: ',col)
        #print( "ANOVA results: F=", f_val, ", P =", p_val)  
        if (f_val <=10):
            cols_that_need_further_investigation.append([col, f_val, p_val])
        elif (f_val > 50):
            cols_with_big_F_score.append([col, f_val, p_val])
        else:
            cols_with_low_F_score.append([col, f_val, p_val])
    cols_that_need_further_investigation.sort(key=lambda x:x[1], reverse=True)
    cols_with_big_F_score.sort(key=lambda x:x[1], reverse=True)
    cols_with_low_F_score.sort(key=lambda x:x[1], reverse=True)
    return cols_that_need_further_investigation, cols_with_big_F_score, cols_with_low_F_score


# column_info is list contains [col_name, f_val, p_val]
def make_boxplots(column_info, data, subplot_size=[4,4,1]):
    r = subplot_size[0]
    c = subplot_size[1]
    i = subplot_size[2]
    plt.figure(figsize=(20,20))
    for col in column_info:
        #plt.figure()
        plt.subplot(r,c,i)
        i+=1
        sns.boxplot(data=data, y=y_full, x=col[0])


def find_n_percent_of_data(total_rows, n=5):
    '''
    total_rows => 100%
     y => n%
     we know n(5get_ipython().run_line_magic(",", " for example), so we find y on this formula:")
     y = n*total_rows/100
    ''' 
    return n * total_rows / 100


def make_dist(data, col_info_list):
    y = find_n_percent_of_data(data.shape[0],5)
    i = 1
    plt.figure(figsize=(20,20))
    for col_info in col_info_list:
        col_name = col_info[0]
        #plt.figure()
        plt.subplot(4,5,i)
        i += 1
        sns.histplot(data[col_name])
        plt.xticks(rotation=45)
        plt.axhline(y,color='r')


cols_that_need_further_investigation, cols_with_big_F_score, cols_with_low_F_score = score_ANOVA(without_nan_df)


cols_that_need_further_investigation


make_boxplots(cols_that_need_further_investigation,data=without_nan_df)


all_rows = Full_train.shape[0] 
for col_info in cols_that_need_further_investigation:
    # get value_counts in percentage
    col_name = col_info[0]
    print(Full_train[col_name].value_counts() / all_rows * 100)


invest_df = Full_train[['ExterCond', 'LotConfig', 'SalePrice']]
group = invest_df.groupby(['ExterCond', 'LotConfig'],as_index=False).mean()
pivot = group.pivot(columns='ExterCond', index='LotConfig')
pivot
#f_score, p_val = stats.f_oneway()


sns.heatmap(pivot.fillna(0))


invest_df['ExterCond'].value_counts()


group = invest_df.groupby(['ExterCond'],as_index=False)
group.mean()


stats.f_oneway(group.get_group('Ex')['SalePrice'], group.get_group('Fa')['SalePrice'], group.get_group('Gd')['SalePrice'])


invest_df['LotConfig'].value_counts()


group = invest_df.groupby(['LotConfig'],as_index=False)
group.mean()


stats.f_oneway(group.get_group('Corner')['SalePrice'], group.get_group('Inside')['SalePrice'], group.get_group('CulDSac')['SalePrice'])


cols_with_big_F_score


make_boxplots(cols_with_big_F_score, without_nan_df)


without_nan_df['HeatingQC'].value_counts()


cols_with_big_F_score


cols_with_big_F_score


make_dist(without_nan_df,cols_with_big_F_score)


cols_with_low_F_score


make_boxplots(cols_with_low_F_score, without_nan_df)


make_dist(without_nan_df,cols_with_low_F_score)


for col_info in cols_with_big_F_score+cols_with_low_F_score:
    # get value_counts in percentage
    col_name = col_info[0]
    print(without_nan_df[col_name].value_counts() / len(without_nan_df) * 100)
    print()


cols_to_investigate = ['CentralAir', 'SaleCondition','MSZoning','PavedDrive','SaleType','RoofStyle','BldgType','LandContour']


invest_df = Full_train[cols_to_investigate+['SalePrice']]
invest_df


group = invest_df.groupby(['CentralAir'], as_index=False).mean()
group.head()


for col in cols_to_investigate:
    group = invest_df.groupby([col], as_index=False).mean()
    print(group.head())
    print()





dic = get_dic_name_need_change()
dic


features_cat_no_nan_very_good = [col[0] for col in cols_with_big_F_score]
print(features_cat_no_nan_very_good)


cols_with_big_F_score


features_cat_no_nan_maybe_good = [col[0] for col in cols_with_low_F_score]
print(features_cat_no_nan_maybe_good)


cols_with_low_F_score


df_with_nan = Full_train[low_cardinality_cols_with_nans].copy(deep=True)
df_with_nan = df_with_nan.join(y_full)
df_with_nan


df_with_nan.isna().sum()[df_with_nan.isna().sum() > 0]


Full_train['PoolArea'].value_counts()


Full_train['Electrical'].value_counts()


indicies_elec = Full_train['Electrical'].isna()[Full_train['Electrical'].isna()].index
indicies_elec


Full_train['MasVnrType'].value_counts()


indicies_masvnr = Full_train['MasVnrType'].isna()[Full_train['MasVnrType'].isna()].index
indicies_masvnr


# I added numeric features describing garage to garage_feat to ssee whole picture
garage_feat = ['GarageType','GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageCars', 'GarageArea']
nadf = Full_train[garage_feat].isna()


Full_train[garage_feat]


(Full_train['GarageCars'] == 0).sum()


(Full_train['GarageArea'] == 0).sum()


Full_train['GarageYrBlt'].isna().sum()


Full_train[garage_feat][(nadf == True).any(axis=1)].head(5)


bsm_feat = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2', 
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
nadf = Full_train[bsm_feat].isna()
Full_train[bsm_feat][(nadf == True).any(axis=1)].head(5)


indices = [333,949]


Full_train[bsm_feat].loc[indices]


indices_to_drop = list(indicies_masvnr) + indices + list(indicies_elec)
print(len(indices_to_drop))
print(indices_to_drop)


df_with_nan.loc[indices_to_drop]


df_replaced_nan = df_with_nan.replace(np.nan, 'None').drop(indices_to_drop)
df_replaced_nan


cols_that_need_further_investigation, cols_with_big_F_score, cols_with_low_F_score = score_ANOVA(df_replaced_nan)


print(cols_with_big_F_score)


make_boxplots(cols_with_big_F_score, df_replaced_nan)


make_dist(df_replaced_nan, cols_with_big_F_score)


print(cols_with_low_F_score)


make_boxplots(cols_with_low_F_score,df_replaced_nan)


make_dist(df_replaced_nan, cols_with_low_F_score)


for col_info in cols_with_big_F_score+cols_with_low_F_score:
    # get value_counts in percentage
    col_name = col_info[0]
    print(df_replaced_nan[col_name].value_counts() / len(df_replaced_nan) * 100)
    print()


print(cols_that_need_further_investigation)


make_boxplots(cols_that_need_further_investigation,df_replaced_nan)


df_group = df_replaced_nan[['MiscFeature', 'SalePrice']]
group = df_group.groupby('MiscFeature', as_index=False)


stats.f_oneway(group.get_group('Othr')['SalePrice'], group.get_group('Shed')['SalePrice'], group.get_group('Gar2')['SalePrice'])


features_cat_from_nan_very_good = [col[0] for col in cols_with_big_F_score]
print(features_cat_from_nan_very_good)


features_cat_from_nan_maybe_good = [col[0] for col in cols_with_low_F_score]
print(features_cat_from_nan_maybe_good)


cols_with_low_F_score


print(features_cat_from_nan_very_good)


print(features_cat_no_nan_very_good)


print(features_num_good_corr)


print(features_cat_from_nan_maybe_good)


print(features_cat_no_nan_maybe_good)


X_full['Total_bath'] = X_full['FullBath'] + X_full['HalfBath']





from itertools import combinations, permutations


c = combinations([1,2,3],2)
for i in c:
    print(i)


X_full.select_dtypes(exclude='object').columns


X_full['Total_bath'] = X_full['FullBath'] + X_full['HalfBath']


selected_features_df = X_full.select_dtypes(exclude='object')


total_columns = selected_features_df.columns
c = combinations(total_columns,2)
for comb in c:
    x1 = comb[0]
    x2 = comb[1]
    selected_features_df[[x1 + '_x_' + x2]] = selected_features_df[x1] * selected_features_df[x2]
selected_features_df = selected_features_df.join(y_full)


X_full.isna().sum()[X_full.isna().sum() > 0]


X_full.select_dtypes(exclude='object')


cross_features_df = selected_features_df.iloc[:, X_full.select_dtypes(exclude='object').shape[1]:]
cross_features_df


cross_features_df = cross_features_df.dropna()


#cols_that_need_further_investigation, cols_with_big_F_score, cols_with_low_F_score  = score_ANOVA(cross_features_df)


#cols_with_big_F_score


#make_boxplots(cols_with_big_F_score[:15],data=cross_features_df,subplot_size=[6,6,1])


#cols_with_low_F_score[:15]


#s = cols_with_low_F_score


#cols_with_low_F_score[:15]


#make_boxplots(s[:10],data=cross_features_df,subplot_size=[6,6,1])


cross_corr = cross_features_df.corr()['SalePrice'].sort_values(ascending=False).dropna()


cross_corr[cross_corr > 0.75]


cross_cols = cross_corr[cross_corr > 0.5].iloc[1:].index


new_Full_df = selected_features_df.join(X_full.select_dtypes(include='object'))
new_Full_df


print(features_cat_from_nan_very_good)


print(features_cat_no_nan_very_good)


print(features_num_good_corr)


print(features_cat_from_nan_maybe_good)


print(features_cat_no_nan_maybe_good)


new_Full_df['Total_Bath'] = new_Full_df['FullBath'] + new_Full_df['HalfBath']


new_train_df = new_Full_df[['SalePrice'] + features_cat_from_nan_very_good + features_cat_no_nan_very_good + 
           features_num_good_corr + features_cat_from_nan_maybe_good + features_cat_no_nan_maybe_good +
           list(cross_cols)
           ]


new_train_df



