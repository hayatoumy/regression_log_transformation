import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 

pd.set_option('max_columns', 100)

training = pd.read_csv('train.csv')
print('training shape', training.shape)

training.columns = [co.lower().replace(' ','_').replace('/','_') for co in training.columns]

# changing NA to 'NA' in the variables with str value named NA
subset_training = ['alley','bsmt_qual','bsmt_cond','bsmt_exposure','bsmtfin_type_1','bsmtfin_type_2',\
                   'fireplace_qu','garage_type' ,'garage_finish' ,'garage_qual' ,'garage_cond' ,'pool_qc' ,'fence'\
                   ,'misc_feature']
[training[x].fillna('NA', inplace = True) for x in subset_training] # run only once! commented to avoid problems

num_null = list(training.loc[:, training.isnull().sum() !=0].columns)
print(num_null)

# dropping the one column that has object type in the list
num_null.pop(1)

# replacing null with zero in numeric columns
[training[x].fillna(0, inplace = True) for x in num_null]

# replacing null values with None in the object column
training['mas_vnr_type'] = training['mas_vnr_type'].fillna('None')

print('which ones still have nulls')
training.loc[:,training.isnull().sum() != 0].head(2)

# seeing dictionary http://jse.amstat.org/v19n3/decock/DataDocumentation.txt and searching for 'ordinal'
# the ordinals are:
ordinals = ['lot_shape', 'utilities', 'land_slope', 'exter_qual', 'exter_cond', 'bsmt_cond', 'bsmt_exposure', 'bsmtfin_type_2', 'heating_qc', 'electrical', 'kitchen_qual', 'functional', 'fireplace_qu', 'garage_finish', 'garage_qual', 'garage_cond', 'paved_drive','pool_qc', 'fence']

test = pd.read_csv('test.csv')

# encoding ordinal variables
new_training = pd.DataFrame(training[ordinals])
training[ordinals] = pd.DataFrame({col: new_training[col].astype('category').cat.codes for col in new_training}, index = training.index)

print('sale price averages, by condition_2:')
training['saleprice'].groupby(training['condition_2']).mean().round(3).sort_values(ascending = False)

print('the dataframe of columns withouth null values:')
training.loc[:,training.isnull().sum() != 0].shape

now_nulls = list(training.loc[:,training.isnull().sum() != 0].columns)

#since there is too many missing values, I'm gonna drop these columns entirely as I don't see much correlation with 
# y anyway
to_drop = list(training.loc[:,training.isnull().sum() != 0].columns)
training = training.drop(to_drop, axis = 1)
print('now we have training df of size: ', training.shape)

#--------------------------------------------
### CLEANING TEST DATA NOW, repating steps done for training data:

test = pd.read_csv('test.csv')

test.columns = [co.lower().replace(' ','_').replace('/','_') for co in test.columns]
[test[x].fillna('NA', inplace = True) for x in subset_training] # run only once! commented to avoid problems
[test[x].fillna(0, inplace = True) for x in num_null]
test['mas_vnr_type'] = test['mas_vnr_type'].fillna('None')

new_test = pd.DataFrame(test[ordinals])
test[ordinals] = pd.DataFrame({col: new_test[col].astype('category').cat.codes for col in new_test}, index = test.index)

test = test.drop(to_drop, axis = 1)
print('test dataframe shape is now: ', test.shape)

#-----------------------------------------------
### SOME EDA    

filtered = pd.DataFrame(training.corrwith(training['saleprice']))
filtered['abs'] = abs(filtered[0])
print(filtered.sort_values(by = 'abs', ascending  = False).head(9)) # above 0.6 correlation

highest_corrs = list(filtered.sort_values(by = 'abs', ascending  = False).index[:9])
print(highest_corrs)

plt.figure(figsize=(15, 15))
sns.heatmap(pd.DataFrame(filtered[0]), annot = True, vmin = -1, vmax = 1, cmap = 'coolwarm');

plt.hist(training['saleprice']);
plt.title('Sale Prices in Training Set');
# this tells me I need to log-transform y 

#------------------------------------------------------------------------
### PREPARING THE MODEL:

from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression, ElasticNetCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split

# fetching the numerical columns only, that are not my target
features = [col for col in training._get_numeric_data().columns if col != 'saleprice']
X = training[features]
y = training['saleprice']

# doing the same thing for our test dataset to match dimensions with X
variables = [col for col in test._get_numeric_data().columns]
test_data = test[variables]
test_data.shape
print("X df shape {}, test df shape {}".format(X.shape, test_data.shape))

##### TEST/TRAIN/SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# scaling
ss = StandardScaler()
X_train_sc = ss.fit_transform(X_train)
X_test_sc = ss.transform(X_test)

#--------------------------------------------------------------------
### POWER TRANSFORMER:

# transforming the y:
pt_y = PowerTransformer() # includes rescaling
pt_y.fit(y_train.to_frame())
y_train_pt = pt_y.transform(y_train.to_frame()) 
y_test_pt = pt_y.transform(y_test.to_frame())

plt.hist(y_train_pt,bins=25);
plt.title('Sale Prices After Log-Transformation');

# instantiate the model
lr = LinearRegression()
lasso = LassoCV(cv = 5)
ridge = RidgeCV()
elastic = ElasticNetCV()

print('scoring the model: ')
print(cross_val_score(lr, X_train, y_train_pt, cv = 5).mean())
print(cross_val_score(lasso, X_train_sc, y_train_pt[:,0], cv = 5).mean())
print(cross_val_score(ridge, X_train_sc, y_train_pt[:,0], cv = 5).mean())
print(cross_val_score(elastic, X_train_sc, y_train_pt[:,0], cv = 5).mean())

# model fitting and evaluation:
ridge.fit(X_train_sc, y_train_pt);
print('ridge score on training set:', ridge.score(X_train_sc, y_train_pt))
print('ridge score on test set: ', ridge.score(X_test_sc, y_test_pt))

# predicting:
ridge_pred = ridge.predict(X_test_sc)

plt.hist(ridge_pred);
plt.title('ridge predictions, based on log-transformation'.title());

# to go back to originals:
# The .reshape(-1,1) method changes a numpy array into a numpy matrix with 1 column
ridge_pred_reversed = pt_y.inverse_transform(ridge_pred.reshape(-1,1))

plt.hist(ridge_pred_reversed);
plt.title('ridge predictions, back to original values'.title());

print('ridge score on target: ', r2_score(y_test, ridge_pred_reversed))

resid = y_test_pt - ridge_pred
plt.hist(resid);
plt.title('errors distribution of ridge prediction'.title());


test_data_sc = ss.transform(test_data)
saleprice = ridge.predict(test_data_sc)
plt.hist(saleprice); #after rescaling
plt.title('sale prices after log transformation'.title());

saleprice = pt_y.inverse_transform(saleprice)
plt.hist(saleprice); # after un-rescaling (going back to originals)
plt.title('sale prices going back to original values'.title());

saleprice = [i[0] for i in saleprice] # it was a list of lists
pd.DataFrame({'SalePrice' : saleprice}, index = test['id']).to_csv('second_prediction.csv')
pd.read_csv('second_prediction.csv').head()

#---------------------------------------------------------
### Getting Coefficients: 
ridge_coefs = pd.DataFrame(ridge.coef_, columns = X_train.columns).T
ridge_coefs['originals'] = np.exp(ridge_coefs[0]) #getting original values, since it's log-transformed
ridge_coefs['abs_vals'] = ridge_coefs['originals'].abs()
ridge_coefs.sort_values(by = 'abs_vals', ascending = False)


#-------------------------------------------------------------
### plotting Training set SalePrices by neighborhood
plt.figure(figsize = (12,12))
plt.barh(training['neighborhood'], training['saleprice']);
#remove the %matplotlib inline from the top of the document first in order to be able to save!
plt.savefig('saleprice_by_neighborhood.png')


##P.S. no need to use GridSearch with RidgeCV because that latter has a parameter within it to test a few alphas, which 
#I can change. I could use GridSearch with Ridge however. 
#===================================================================================================
