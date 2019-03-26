### The Problem
We have a dataset of features of houses, and want to predict the house sale price based on those features.

### The Dataset
We have a training dataset, and a testing dataset. The variables are a mix of numerical, ordinal, categorical, and string. Many of which had missing values. <br />
I started by loading, and exploring the training dataset. I changed the mistakenly null values into 'NA' strings in the corresponding categorical variables. 
<br /> Then based on the dictionary of the variables, found at: http://jse.amstat.org/v19n3/decock/DataDocumentation.txt and https://www.kaggle.com/c/dsi-us-7-project-2-regression-challenge/data I encoded some ordinal variables into numbers to incorporate them in my models. 

### EDA findings 
The strongest correlations with SalePrice, were:
overall_qual	0.800207	
gr_liv_area	0.697038	
exter_qual	-0.662321	
garage_area	0.649897	
garage_cars	0.647781	
kitchen_qual	-0.641416	
total_bsmt_sf	0.629303
1st_flr_sf	0.618486
<br /> Note about this: I'm sure there are other important factors too, maybe more important, like the neighbourhood, utilities, central air, etc. But I either couldn't come up with an efficient way to encode them without masking the underlying meaning to them; or I reasoned they can't be more important, so I could avoid overfitting, and reduce the complexity of the model. 
<br /> **important note** The exterior quality and the kitchen quality are NOT negatively correlated, they are positively correlated, but when I encoded them, Python treated the first entry as 1, which corresponds to Excellent, up to 5 which corresponds to Poor. That is just the way the string values are entered to describe the quality. 

SalePrice is skewed to the right, suggesting I need to make a log transformation on it, to it would become as close to normal as possible; therefore so would do predictions. I applied reverse transformation though to readily interpret coefficients. 

### Modeling Methods
Split my training set using 5-folds cross-validation to train the model, and test it. <br />
I then found the $R^2$ score to measure goodness of fit to the hold out set. 

### Modles Tried
Linear regression with all the cleaned, encoded features, with regularization applied to it. This gave acceptable predictions, but not the best. 
<br /> Then I log-transformed the target variable, again applied regularizations on it. Ridge proved best, with $R^2 \approx 0.894$. Errors were almost normal, centered at zero, with a couple of outliers. 


Interactions and Polynomial Features: I then selected only the variables with the highest correlations, mentioned above, found the interactions between them, squared each of them, and added the new modified features to the data, then fit a linear regression on them, which did better than Ridge, Lasso, or Elastic Net. This was not as good as the previous model, but close. 


Then I tried using the same highly correlated variables, with a log-transformed SalePrice, the model performed equal to the last one, or worse. So I didn't continue with it. 


If I had more time, I would try a combination of the above techniques, with new set of variables. Then I would look into the possibilty of encoding more ordinal, or nominal variables if logically possible. <br />
I am comfortable with my models because they are logical, the skewed target variable SalePrice calls for log-transformation, the Ridge controls bias, which is more important to model's accuracy than variance, if I must choose between a slight trade-off. <br />
I would also look more into errors, see if they have a discernable pattern, which suggests autocorrelation, or multicollinearity, or both. Then I would examine correlations between the predictors, and if any two of them has linear correlation above 0.8, I would discard one of them from the model. 

### Recommendations: 
- The features appear to add most value are the ones highly positively correlated with SalePrice. The bigger the house, garage, living area, the more expensive the house. 
<br /> The better the quality of the house kitchen, exterior, and overall; the pricier the house. Also the worse those are, the cheaper the house would sell. 
- Homeowners can revamp and fix their homes to sell for more, to enhance the quality and condition of exterior, kitchen, and overall. 
- Top 5 highest prices average, by neighborhoor, came out to be:
neighborhood
StoneBr    329675.737
NridgHt    322831.352
NoRidge    316294.125
GrnHill    280000.000
Veenker    253570.588
Therefore, I would say, these neighbourhoods are good investment. 
- I think this model can be generalized to other cities, there's no variables I thought specific to Aimes, Iowa. That is because I didn't incorporate the neighbourhoods in my model. 
