#!/usr/bin/env python
# coding: utf-8
Housing price in Washington state
A data set for the sale price of houses and the basic information of the house in Washington, USA.
Based on the data analysis performace, preditc the factors majorly affected the trend of house price.
# In[ ]:


Importing the Dataset


# In[2]:


# Useful imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
output_notebook()
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
WA = pd.read_csv('WAhousingdata.csv')
#https://www.kaggle.com/shree1992/housedata#data.csv
#WA.columns = WAdata.feature_names
WA.head()
#WAdata = sklearn.datasets.load_WA()
#print(WAdata.DESCR)
print(WA.head())
WA.info()

Check the dataset type
# In[3]:


WA = pd.read_csv('WAhousingdata.csv')
WA.head()
print(WA)# Load the WA Housing dataset
price = WA['price']
features = WA.drop('price', axis = 1)
print("Washington housing dataset has {} data points with {} variables each.".format(*WA.shape)) 
WA.shape

Visulization Dataset
# In[4]:


WA = pd.read_csv('WAhousingdata.csv')
WA.plot(kind="scatter", x="condition", y="yr_built", alpha=0.4, figsize=(10,7),
    c="price", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
#Color code from the most expensive to the cheapest area
#Create a scatter plot with house of year built and house condition to visualize the data

Data format processing
# In[5]:


indexNames = WA[WA['price'] == 0].index
WA.drop(indexNames, inplace = True)
WA.describe()
#Describe the Washinton house datasetss in count, mean, min and max pices rows.
#The highest price would be $26,590,000, the lowest price would be $7,8000.
#The std shows the standard deviation in the 25%, 50% and 75% rows show the corresponding percentiles

Check data type and Skewness
# In[6]:


WA.info()
WA.isnull().sum()
sns.distplot(WA['price'])
plt.figure(figsize = (5,5))
print("skew: ",WA.price.skew())

Histogram graphs of numeric variables
# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
WA.hist(bins=50, figsize=(20,15))
plt.savefig("attribute_histogram_plots")
plt.show()

Detect correlation between numerical features and variable Price
# In[8]:


#* Generate a scatter plot of sqft_living (y-axis) and price (x-axis)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
scatter_plot= plt.figure(figsize=(5,5))
axis = scatter_plot.add_subplot(1,1,1)
axis.set_title('Scatterplot of price vs sqft_living')
axis.scatter(WA['sqft_living'],WA['price'])
axis.set_xlabel('price')
axis.set_ylabel('sqft_living')
plt.show()


# In[9]:


scatter_plot= plt.figure(figsize=(5,5))
axis = scatter_plot.add_subplot(1,1,1)
axis.scatter(WA['bedrooms'],WA['price'])
axis.set_title('Scatterplot of price vs condition')
axis.set_xlabel('bedrooms')
axis.set_ylabel('price')
scatter_plot.show()
sns.lmplot(x='bedrooms', y='price', data=WA)
plt.show()


# In[10]:


# WA.plot(kind="scatter", x="condition", y="price", alpha=0.5)
plt.savefig('scatter.png')
sns.lmplot(x='price', y='condition', data=WA)
plt.show()


# In[11]:


corr_matrix = WA.corr()
#coefficients close to zero indicate that there is no linear correlation.
corr_matrix["price"].sort_values(ascending=False)


# In[12]:


#* Generate the plots using matplotlib and seaborn
sns.lmplot(x='price', y='sqft_living', data=WA)
plt.show()
WA['sqft_living'] = WA['price']/WA['sqft_living']
corr_matrix = WA.corr()
corr_matrix["price"].sort_values(ascending=False)
#the correlation between sqft of living variable and the house price

Building a house price Linear Regression forecasting model Create training sets and test sets
# In[13]:


from sklearn.model_selection import train_test_split
#import train_test_split
# Split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(features,price,test_size=0.2,random_state=10)
print(features.shape)
print(X_train.shape)
print(X_test.shape)


# In[14]:


#Constructing feature vectors and target vectors
features = ['sqft_living',
            'sqft_lot',
            'view',
            'bathrooms',
            'condition',
            'bedrooms',
            'sqft_above',
            'sqft_basement',
            "yr_built",
            "yr_renovated"
           ]
X = WA[features]
Y = np.log(WA['price'])
print(X.shape,Y.shape)


# In[17]:


#Feature preprocessing
X = WA[['bathrooms', 'bedrooms', 'sqft_living', 'sqft_lot', 'condition', 'yr_built', 
         'view', 'yr_renovated']]
Y = WA['price']
#Train and Build Linear Regression Model
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
lm = linear_model.LinearRegression()
model = lm.fit(X_train,Y_train)
print(model.intercept_,model.coef_)

Evaluation model
# In[18]:


#A higher R-square value means a better fit
Y_predict = regressor.predict(X_test)#Linear Regression R squared
print('Linear Regression R squared": %.4f' % regressor.score(X_test, Y_test))


# In[19]:


import numpy as np
#predict in Linear Regression MAE 
from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(Y_predict, Y_test)
lin_rmse = np.sqrt(lin_mse)
print('Linear Regression RMSE: %.4f' % lin_rmse)
#the value of every house in the test set within $616071 of the real price


# In[ ]:




