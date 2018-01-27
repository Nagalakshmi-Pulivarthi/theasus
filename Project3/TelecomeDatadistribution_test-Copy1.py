
# coding: utf-8

# In[10]:

# Importing Dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib
matplotlib.style.use('ggplot') 


# In[11]:

# 
file='TelecomUsageDemogone.csv'
total_data=pd.read_csv(file)
# data=['TENURE','TOTALCHARGES','MONTHLYCHARGES','MONTHLY_MINUTES_OF_USE','TOTAL_MINUTES_OF_USE','MONTHLY_SMS','TOTAL_SMS']
data=['TENURE','TOTALCHARGES','MONTHLYCHARGES','MONTHLY_MINUTES_OF_USE','MONTHLY_SMS','TOTAL_SMS',"TOTAL_MINUTES_OF_USE"]


# In[12]:

total_data.isnull().values.any()


# In[13]:

# Continous Feature Distribution
telecome_data=pd.read_csv(file,usecols=data )
#telecome_data.plot(kind='hist',subplots=True,range=(0,150),bins=100,figsize=(10,10))
                      # Set figure size

telecome_data.hist(bins=100,figsize=(10,10))
plt.show()


# In[14]:

total_data.head()
# del total_data['CUSTOMERID']


# # Correltion  refers to a mutual relationship or accociation between  quantities

# In[55]:

def plot_corr(total_data, size=10):
    corr = total_data.corr()  
#     cmap = cm.get_cmap('jet', 30)
    # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
#     cax = ax.imshow(corr, interpolation="nearest", cmap=cmap)
    ax.matshow(corr)   # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
#     fig.colorbar()
    plt.show()
#     plt.tight_layout()


# In[56]:

plot_corr(total_data)



# In[57]:

total_data.corr()


# In[58]:

# import seaborn as sns

# f, ax = plt.subplots(figsize=(10, 10))
# corr = total_data.corr()
# sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(250, 10, as_cmap=True),
#             square=True, ax=ax)
# plt.show()


# In[422]:

# # plotting Corealtion Matrix
# import seaborn as sns

# f, ax = plt.subplots(figsize=(10, 8))
# corr = telecome_data.corr()
# sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
#             square=True, ax=ax)
# plt.show()


# In[386]:

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[387]:

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

categorical_data=['CHURN','MONTHLYCHARGES','TOTALCHARGES','GENDER','TENURE','PHONESERVICE','CONTRACT','MULTIPLELINES','PARTNER']
total_data.head()


# In[388]:

telecome_data=pd.read_csv(file,usecols=categorical_data )
telecome_data.head()
# telecome_data1['PHONESER


# In[389]:

np.random.seed(sum(map(ord, "categorical")))


# In[390]:

ydata=telecome_data['MONTHLYCHARGES']


# In[391]:

sns.set(style="ticks")

flatui = ["#9b59b6", "#3498db", "#95a5a6"]

# This Function takes as input a custom palette
g = sns.barplot(x="GENDER", y="TOTALCHARGES", hue="CHURN",
    palette=sns.color_palette(flatui),data=telecome_data,ci=None)

# remove the top and right line in graph
sns.despine()

# Set the size of the graph from here
g.figure.set_size_inches(12,7)
# Set the Title of the graph from here
g.axes.set_title('Do We tend to \nTip high on Weekends?',
    fontsize=34,color="b",alpha=0.3)
# Set the xlabel of the graph from here
g.set_xlabel("Gender",size = 67,color="g",alpha=0.5)
# Set the ylabel of the graph from here
g.set_ylabel("TotalCharges",size = 67,color="r",alpha=0.5)
# Set the ticklabel size and color of the graph from here
g.tick_params(labelsize=14,labelcolor="black")
plt.show()


# In[392]:

sns.set(style="ticks")

flatui = ["#9b59b6", "#3498db",]

# This Function takes as input a custom palette
g = sns.barplot(x="CHURN", y="TOTALCHARGES", hue="PHONESERVICE",
    palette=sns.color_palette(flatui),data=telecome_data,ci=None)

# remove the top and right line in graph
sns.despine()

# Set the size of the graph from here
g.figure.set_size_inches(12,7)
# Set the Title of the graph from here
g.axes.set_title('Do We tend to \nTip high on Weekends?',
    fontsize=34,color="b",alpha=0.3)
# Set the xlabel of the graph from here
g.set_xlabel("CHURN",size = 67,color="g",alpha=0.5)
# Set the ylabel of the graph from here
g.set_ylabel("TotalCharges",size = 67,color="r",alpha=0.5)
# Set the ticklabel size and color of the graph from here
g.tick_params(labelsize=14,labelcolor="black")
plt.show()


# In[393]:

sns.set(style="ticks")

flatui = ["#9b59b6", "#3498db","#2ecc71"]

# This Function takes as input a custom palette
g = sns.barplot(x="CHURN", y="TOTALCHARGES", hue="PARTNER",
    palette=sns.color_palette(flatui),data=telecome_data,ci=None)

# remove the top and right line in graph
sns.despine()

# Set the size of the graph from here
g.figure.set_size_inches(12,7)
# Set the Title of the graph from here
g.axes.set_title('Do We tend to \nTip high on Weekends?',
    fontsize=34,color="b",alpha=0.3)
# Set the xlabel of the graph from here
g.set_xlabel("CHURN",size = 67,color="g",alpha=0.5)
# Set the ylabel of the graph from here
g.set_ylabel("TotalCharges",size = 67,color="r",alpha=0.5)
# Set the ticklabel size and color of the graph from here
g.tick_params(labelsize=14,labelcolor="black")
plt.show()


# In[ ]:




# In[ ]:



