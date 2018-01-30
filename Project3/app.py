#import dependencies
import csv
from flask import send_file
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import cm
import json
from flask import Flask, render_template,jsonify,session 
import datetime as dt
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
# Import Scikit Learn
from sklearn.linear_model import LogisticRegression
# Import Pickle
import _pickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from flask import request  
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot') 
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def report2dict(cr):
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)
    
    # Store in dictionary
    measures = tmp[0]

    D_class_data = []
    for row in tmp[1:]:
        class_label = row[0]
       # print(class_label)
        d ={}
        d["class_label"] = class_label
        for j, m in enumerate(measures):
            d[m.strip()] = row[j+1]
          
        D_class_data.append(d)
    return D_class_data
# ===========================Flask Connection==========================
app = Flask(__name__)

@app.route('/')
# Return the dashboard homepage.
def index():
    DataObject = {}
    sns.set_palette(palette=None)
    sns.set(style="ticks")
    file='TelecomUsageDemoone.csv'
    total_data=pd.read_csv(file)
    data=['TENURE','TOTALCHARGES','MONTHLYCHARGES','MONTHLY_MINUTES_OF_USE','MONTHLY_SMS','TOTAL_SMS',"TOTAL_MINUTES_OF_USE","CHURN"]

    hasAnyNullValues = total_data.isnull().values.any()
    # Continous Feature Distribution
    telecome_data=pd.read_csv(file,usecols=data )
    #telecome_data.plot(kind='hist',subplots=True,range=(0,150),bins=100,figsize=(10,10))
    # Set figure size

    basePath ="static/img/"
    telecome_data.hist(bins=100,figsize=(10,10),color="#00008B")
    histImageUrl = basePath + "histogram.jpg"
    plt.savefig(histImageUrl)

    DataObject["hasAnyNullValues"] =hasAnyNullValues
    DataObject["histImageUrl"] = histImageUrl

    #print(DataObject)

    # # Correaltion matrix plot
    def plot_corr(total_data, corrFigpath,size=11):
        corr = total_data.corr()  
        cmap = cm.get_cmap('viridis',30)
        # data frame correlation function
        fig, ax = plt.subplots(figsize=(size, size))
    #     cax = ax.imshow(corr, interpolation="nearest", cmap=cmap)
        print("corr:")
        print(corr)
        ax.matshow(corr, interpolation="nearest", cmap=cmap)   # color code the rectangles by correlation value
        plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
        plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
        plt.savefig(corrFigpath)
    #     fig.colorbar()

    corrFigpath=basePath+"correalation.jpg"
    plot_corr(total_data,corrFigpath)

    total_data.corr()
    del total_data['TOTAL_MINUTES_OF_USE']
    correaltion=plot_corr(total_data,corrFigpath)

    DataObject["correaltion"]=correaltion
    DataObject["corrFigpath"]=corrFigpath

    #categorical Distribution
    np.random.seed(sum(map(ord, "categorical")))
    file1='TelecomUsageDemoone.csv'
    all_data=pd.read_csv(file1)
    categorical_data=['CHURN','MONTHLYCHARGES','TOTALCHARGES','GENDER','TENURE','PHONESERVICE','CONTRACT','MULTIPLELINES','PARTNER']
    all_data=pd.read_csv(file,usecols=categorical_data)
    
    
    flatui = ["yellow", "#00008B", "#32CD32"]

    # This Function takes as input a custom palette
    #g = sns.factorplot(x="CHURN", y="TOTALCHARGES", hue="MULTIPLELINES",kind="bar",col="GENDER",
    #palette=sns.color_palette(flatui),data=all_data,ci=None)
    g = sns.factorplot(x="CHURN", y="TOTALCHARGES", hue="PHONESERVICE",kind="bar",col="GENDER",palette=sns.color_palette(flatui),data=all_data,ci=None)
    # # remove the top and right line in graph
    sns.despine()
 
    barImageUrl = basePath + "barchart.jpg"
    # barImageUrl1 = basePath + "barchart1.jpg"
    plt.savefig(barImageUrl)
    DataObject["barImageUrl"]=barImageUrl

    g = sns.factorplot(x="CHURN", y="TOTALCHARGES", hue="MULTIPLELINES",kind="bar",col="GENDER", palette=sns.color_palette(flatui),data=all_data,ci=None)
    barImageUrl1=basePath + "barchart1.jpg"
    plt.savefig(barImageUrl1)
    DataObject["barImageUrl1"]=barImageUrl1
    # plt.savefig(barImageUrl1)
    # DataObject["barImageUrl1"]=barImageUrl1

    #plt.show()
    file='TelecomUsageDemogFinal.csv'
    total_data=pd.read_csv(file)
    X=total_data.drop(["CHURN","GENDER","PHONESERVICE","MULTIPLELINES_No","MULTIPLELINES_No phone service"],1)
    y=total_data["CHURN"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    # Score the Model
    train_score = classifier.score(X_train, y_train)
    test_score = classifier.score(X_test, y_test)

    DataObject["train_score"]=train_score
    DataObject["test_score"]=test_score

    
    auc_score_train= accuracy_score(y_train,classifier.predict(X_train))
    auc_score_test= accuracy_score(y_test,classifier.predict(X_test))

    DataObject["auc_score_train"]=auc_score_train
    DataObject["auc_score_test"]=auc_score_test
    
    logit_roc_auc=roc_auc_score(y_test,classifier.predict(X_test))
    cls_report = classification_report(y_test,classifier.predict(X_test))

    DataObject["logit_roc_auc"]=  logit_roc_auc
    DataObject["cls_report"]=  report2dict(cls_report)

    #print("Logistic AUC=%2.2f" % logit_roc_auc)
    #print(classification_report(y_test,classifier.predict(X_test)))

    from sklearn.metrics import roc_curve
    fpr,tpr,thresholds=roc_curve(y_test,classifier.predict_proba(X_test)[:,1])


    plt.figure()
    plt.plot(fpr,tpr,label="ROC curve (area=%0.2f)"%logit_roc_auc)
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Tru Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.show()

    rocCurveImageUrl = basePath + "roc_curve.jpg"
    plt.savefig(rocCurveImageUrl)
    DataObject["rocCurveImageUrl"]=  rocCurveImageUrl

    # Pickle 
    pickle.dump(classifier, open("Classifier.sav", 'wb'))

    return render_template('index.html', **DataObject)

@app.route('/upload', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return('No file part')

 #
 
    print("files length ")
    file = request.files['file']
    

    replayData=pd.read_csv(file)
    X=replayData.drop(["CHURN"],1)
    DataObject = {}

    # Reload the classifier
    classifier = pickle.load(open("Classifier.sav", 'rb'))
     # Filter it down using the Classifier
    churndata = []
    X.head()
    for index, row in X.iterrows():
        # print(row["SENIORCITIZEN"])
        if(classifier.predict([row['SENIORCITIZEN'],  \
                            row['PARTNER'], \
                            row['DEPENDENTS'], \
                            row['TENURE'], \
                            row['PAPERLESSBILLING'], \
                            row['MONTHLYCHARGES'], \
                            row['TOTALCHARGES'], \
                            row['MONTHLY_MINUTES_OF_USE'], \
                            row['TOTAL_MINUTES_OF_USE'], \
                            row['MONTHLY_SMS'], \
                            row['TOTAL_SMS'], \
                            row['MULTIPLELINES_Yes'], \
                            row['INTERNETSERVICE_DSL'], \
                            row['INTERNETSERVICE_Fiber optic'], \
                            row['INTERNETSERVICE_No'],
                            row['ONLINESECURITY_No'],
                            row['ONLINESECURITY_No internet service'],
                            row['ONLINESECURITY_Yes'],
                            row['ONLINEBACKUP_No'],
                            row['ONLINEBACKUP_No internet service'],
                            row['ONLINEBACKUP_Yes'],
                            row['DEVICEPROTECTION_No'],
                            row['DEVICEPROTECTION_No internet service'],
                            row['DEVICEPROTECTION_Yes'],
                            row['TECHSUPPORT_No'],
                            row['TECHSUPPORT_No internet service'],
                            row['TECHSUPPORT_Yes'],
                            row['STREAMINGTV_No'],
                            row['STREAMINGTV_No internet service'],
                            row['STREAMINGTV_Yes'],
                            row['STREAMINGMOVIES_No'],
                            row['STREAMINGMOVIES_No internet service'],
                            row['STREAMINGMOVIES_Yes'],
                            row['CONTRACT_Month-to-month'],
                            row['CONTRACT_One year'],
                            row['CONTRACT_Two year'],
                            row['PAYMENTMETHOD_Bank transfer automatic'],
                            row['PAYMENTMETHOD_Credit card automatic'],
                            row['PAYMENTMETHOD_Electronic check'],
                            row['PAYMENTMETHOD_Mailed check']])==1):
                churndata.append(row)

    DataObject["RecordsReceived"]=len(X)
    DataObject["RecordsProcessed"]=len(X)
    DataObject["Predictedchurncount"]=len(churndata)
    df = pd.DataFrame(churndata)
    #df.index.rename('_index', inplace=True)
    df.to_csv("churndata.csv")
    return(jsonify(DataObject))
@app.route('/getChurnData', methods=['GET'])
def GetChurnData():
    churnrows=[]
    with open('churndata.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            churnrows.append(row)
    return (jsonify(churnrows))

@app.route('/downloadChurnData', methods=['GET'])
def DownloadChurnData():
    return send_file("churndata.csv", as_attachment=True, attachment_filename='dowload-churndata.csv')

if __name__ == "__main__":
    app.run(debug=True)
