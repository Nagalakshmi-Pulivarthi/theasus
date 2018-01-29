
# coding: utf-8

# In[165]:

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# In[166]:

file='TelecomUsageDemogone.csv'
data=['TENURE','TOTALCHARGES','MONTHLYCHARGES']


# In[167]:

telecome_data=pd.read_csv(file,usecols=data )
#telecome_data.plot(kind='hist',subplots=True,range=(0,150),bins=100,figsize=(10,10))
                      # Set figure size

telecome_data.hist(bins=100,figsize=(20,20))
plt.show()


# In[338]:

telecome_data['TOTALCHARGES'].value_counts()
newdata=telecome_data['TOTALCHARGES']


# In[339]:

total_data=pd.read_csv(file )


# In[340]:

del total_data["SENIORCITIZEN"]


# In[341]:

total_data.corr()


# In[342]:

def plot_correlation(total_data, size=10):
    

    corr = total_data.corr()    # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)   # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
    
    fig.colorbar(cax)
    plt.show()


# In[331]:

plot_correlation(total_data)


# In[332]:

telecome_data.corr()


# In[215]:

# def plot_correlations(telecome_data, size=10):
    

#     corr = telecome_data.corr()    # data frame correlation function
#     fig, ax = plt.subplots(figsize=(size, size))
#     ax.matshow(corr)   # color code the rectangles by correlation value
#     plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
#     plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
    
#     fig.colorbar(cax)
#     plt.show()


# In[216]:

plot_correlations(telecome_data)


# In[313]:

categorical_data=['PHONESERVICE','MULTIPLELINES','GENDER','SENIORCITIZEN','INTERNETSERVICE','MONTHLYCHARGES','TENURE','CONTRACT']


# In[314]:

telecome_data1=pd.read_csv(file,usecols=categorical_data )
telecome_data1.head()
# telecome_data1['PHONESERVICE'].value_counts().plot(kind='bar')
# plt.show()


# In[278]:

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[279]:

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)



# In[280]:

np.random.seed(sum(map(ord, "categorical")))


# In[291]:

ydata=telecome_data1['MONTHLYCHARGES']
ydata.head()


# In[290]:

sns.set(style="ticks")

flatui = ["#9b59b6", "#3498db", "#95a5a6"]

# This Function takes as input a custom palette
g = sns.barplot(x="GENDER", y="MONTHLYCHARGES", hue="MULTIPLELINES",
    palette=sns.color_palette(flatui),data=telecome_data1,ci=None)

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
g.set_ylabel("MultipleLines",size = 67,color="r",alpha=0.5)
# Set the ticklabel size and color of the graph from here
g.tick_params(labelsize=14,labelcolor="black")
plt.show()


# In[305]:

sns.set(style="ticks")

flatui = ["#9b59b6", "#3498db",]

# This Function takes as input a custom palette
g = sns.barplot(x="GENDER", y="TENURE", hue="PHONESERVICE",
    palette=sns.color_palette(flatui),data=telecome_data1,ci=None)

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
g.set_ylabel("Phone Service",size = 67,color="r",alpha=0.5)
# Set the ticklabel size and color of the graph from here
g.tick_params(labelsize=14,labelcolor="black")
plt.show()


# In[316]:

sns.set(style="ticks")

flatui = ["#9b59b6", "#3498db","#2ecc71"]

# This Function takes as input a custom palette
g = sns.barplot(x="GENDER", y="TENURE", hue="CONTRACT",
    palette=sns.color_palette(flatui),data=telecome_data1,ci=None)

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
g.set_ylabel("Contract",size = 67,color="r",alpha=0.5)
# Set the ticklabel size and color of the graph from here
g.tick_params(labelsize=14,labelcolor="black")
plt.show()


# In[ ]:




# In[ ]:



