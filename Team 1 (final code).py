from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
import os
import pandas as pd 
for dirname, _, filenames in os.walk("/Users/himanshushekhar/Desktop/aarohi's project/Claims.xlsx"):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
nRowsRead = 1000 
claims=pd.read_excel("/Users/himanshushekhar/Desktop/aarohi's project/Claims.xlsx")

claims.dataframeName = 'df_Clean.csv'
nRow, nCol = claims.shape
print(f'There are {nRow} rows and {nCol} columns')
claims.head(5)       
        
# Deleting first column
claims.drop(["Serial"],inplace=True,axis=1)     
        
# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()    
# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()    
# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

#claims = pd.read_csv("original project data.csv")

plotPerColumnDistribution(claims, 10, 5)
plotCorrelationMatrix(claims, 8)
plotScatterMatrix(claims, 20, 10)
#---------------------------EDA------------------------------------
## Replacing UP with Uttar Pradesh 
claims.loc[(claims.State == "UP"), "State"] = "Uttar Pradesh"

## Replacing claim with Claim
claims.loc[(claims.Purpose == "claim"), "Purpose"] = "Claim"

## Separating hyderbad among two states. like Andhra Pradesh = Hyderbad, Telengana = Hyderabad 1
claims.loc[(claims.State == "Telengana"), "City"] = "Hyderabad 1"


# dropping all duplicates
claims=claims.drop_duplicates(keep="first")
claims.shape

#Imputation of NaN values
claims.isnull().sum()
claims["Claim_Value"].mean() 
claims["Claim_Value"].median() 
claims["Claim_Value"].fillna(7725,inplace=True)

#changing categorical variables
claims_c1 = pd.DataFrame.copy(claims)
claims_c1.drop(['Product_Age','Call_details','Claim_Value'],inplace=True,axis=1) 
claims_c1 = claims_c1.astype('category')
list(claims.columns)
claims_c2 = pd.DataFrame.copy(claims)
claims_c2.drop(['Region', 'State', 'Area', 'City', 'Consumer_profile', 'Product_category', 'Product_type', 'AC_1001_Issue',
                 'AC_1002_Issue', 'AC_1003_Issue', 'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue', 'Service_Centre',
                 'Purchased_from', 'Purpose', 'Fraud'],inplace=True,axis=1)
claims_cat = pd.concat([claims_c1,claims_c2],axis=1)
claims_cat.info()

#creating dummy variables
dum_variables = pd.get_dummies(claims_cat[['Region', 'State', 'Area', 'City', 'Consumer_profile', 'Product_category', 'Product_type', 'AC_1001_Issue',
                 'AC_1002_Issue', 'AC_1003_Issue', 'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue', 'Service_Centre',
                 'Purchased_from', 'Purpose', 'Fraud']],drop_first=True)
claims_cat_nor = pd.concat([dum_variables,claims_c2],axis=1)

#Normalizing the data (continous variables) 
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
scaler = preprocessing.MinMaxScaler()
scaled = scaler.fit_transform(claims_cat_nor[['Product_Age','Call_details','Claim_Value']])

col = list(claims_cat_nor.columns)
claims_cat_nor_dum = np.concatenate((claims_cat_nor.values[:,0:81],scaled),axis=1)
claims_cat_nor_dum = pd.DataFrame(claims_cat_nor_dum,columns=col)
claims_cat_nor_dum

#Blancing Dummy data
pip install imblearn --user
from imblearn.over_sampling import SMOTE
sm_x_nor_dum = pd.DataFrame.copy(claims_cat_nor_dum)
sm_y_nor_dum = sm_x_nor_dum.pop('Fraud_1')
sm = SMOTE(random_state =101)
X_nor_dum_bal, Y_nor_dum_bal = sm.fit_sample(sm_x_nor_dum, sm_y_nor_dum)
X_nor_dum_bal.shape, Y_nor_dum_bal.shape
 
# Labelling the data
claims_lab = pd.DataFrame.copy(claims_cat)
for i in claims_lab.columns:
   if claims_lab[i].dtype.name =='category':
       encoder = preprocessing.LabelEncoder()
       encoder.fit(list(set(claims_lab[i])))
       claims_lab[i] = encoder.transform(claims_lab[i])
claims_lab.info()

#Balancing the data

from imblearn.over_sampling import SMOTE
sm_x = pd.DataFrame.copy(claims_lab)
sm_y = sm_x.pop('Fraud')
sm = SMOTE(random_state =101)
X_bal, Y_bal = sm.fit_sample(sm_x, sm_y)
X_bal.shape, Y_bal.shape

#Spliting Train Test Data
from sklearn.model_selection import train_test_split
X_train_ndb,X_test_ndb,Y_train_ndb,Y_test_ndb = train_test_split(X_nor_dum_bal,Y_nor_dum_bal,test_size=0.33,random_state=101)
X_train_ndb.shape,X_test_ndb.shape,Y_train_ndb.shape,Y_test_ndb.shape

### Building models 

## 1) Random Forest
from sklearn.ensemble import RandomForestClassifier
claims_RF = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")
claims_RF.fit(X_train_ndb,Y_train_ndb)
pred = claims_RF.predict(X_test_ndb)
prr = claims_RF.predict(X_train_ndb)
pd.crosstab(Y_test_ndb,pred)
print("Train Accuracy Balanced Data",np.mean(Y_train_ndb==prr))
print("Test Accuracy Balanced Data",np.mean(Y_test_ndb==pred))
#Train Accuracy Balanced Data 0.9807407407407407
#Test Accuracy Balanced Data 0.9152336448598131
from sklearn.metrics import classification_report
print(classification_report(Y_test_ndb,pred))

## 2) Decision Tree
from sklearn.tree import DecisionTreeClassifier
claims_DT = DecisionTreeClassifier(criterion = 'entropy')
claims_DT.fit(X_train_ndb,Y_train_ndb)
pred = claims_DT.predict(X_test_ndb)
prr = claims_DT.predict(X_train_ndb)
pd.crosstab(Y_test_ndb,pred)
print("Train Accuracy Balanced Data",np.mean(Y_train_ndb==prr))
print("Test Accuracy Balanced Data",np.mean(Y_test_ndb==pred))
#Train Accuracy Balanced Data 0.9907407407407407
#Test Accuracy Balanced Data 0.8878504672897196
from sklearn.metrics import classification_report
print(classification_report(Y_test_ndb,pred))

## 3) Logistic Regression
from sklearn.linear_model import LogisticRegression
claims_LR = LogisticRegression(random_state = 0)
claims_LR.fit(X_train_ndb,Y_train_ndb)
pred = claims_LR.predict(X_test_ndb)
prr = claims_LR.predict(X_train_ndb)
pd.crosstab(Y_test_ndb,pred)
print("Train Accuracy Balanced Data",np.mean(Y_train_ndb==prr))
print("Test Accuracy Balanced Data",np.mean(Y_test_ndb==pred))
#Train Accuracy Balanced Data 0.8703703703703703
#Test Accuracy Balanced Data 0.8317757009345794
from sklearn.metrics import classification_report
print(classification_report(Y_test_ndb,pred))

## 4) SVM
#linear model
from sklearn.svm import SVC
claims_SVM_linear = SVC(kernel = "linear")
claims_SVM_linear.fit(X_train_ndb,Y_train_ndb)
pred = claims_SVM_linear.predict(X_test_ndb)
prr = claims_SVM_linear.predict(X_train_ndb)
pd.crosstab(Y_test_ndb,pred)
print("Train Accuracy Balanced Data",np.mean(Y_train_ndb==prr))
print("Test Accuracy Balanced Data",np.mean(Y_test_ndb==pred))
from sklearn.metrics import classification_report
print(classification_report(Y_test_ndb,pred))

#Poly model
from sklearn.svm import SVC
claims_SVM_poly = SVC(kernel = "poly")
claims_SVM_poly.fit(X_train_ndb,Y_train_ndb)
pred = claims_SVM_poly.predict(X_test_ndb)
prr = claims_SVM_poly.predict(X_train_ndb)
pd.crosstab(Y_test_ndb,pred)
print("Train Accuracy Balanced Data",np.mean(Y_train_ndb==prr))
print("Test Accuracy Balanced Data",np.mean(Y_test_ndb==pred))
from sklearn.metrics import classification_report
print(classification_report(Y_test_ndb,pred))

#rbf model
from sklearn.svm import SVC
claims_SVM_rbf = SVC(kernel = "rbf")
claims_SVM_rbf.fit(X_train_ndb,Y_train_ndb)
pred = claims_SVM_rbf.predict(X_test_ndb)
prr = claims_SVM_rbf.predict(X_train_ndb)
pd.crosstab(Y_test_ndb,pred)
print("Train Accuracy Balanced Data",np.mean(Y_train_ndb==prr))
print("Test Accuracy Balanced Data",np.mean(Y_test_ndb==pred))
from sklearn.metrics import classification_report
print(classification_report(Y_test_ndb,pred))

## 5) KNN
from sklearn.neighbors import KNeighborsClassifier as KNC
neigh = KNC(n_neighbors= 10)
neigh.fit(X_train_ndb,Y_train_ndb)
pred = neigh.predict(X_test_ndb)
prr = neigh.predict(X_train_ndb)
pd.crosstab(Y_test_ndb,pred)
print("Train Accuracy Balanced Data",np.mean(Y_train_ndb==prr))
print("Test Accuracy Balanced Data",np.mean(Y_test_ndb==pred))
from sklearn.metrics import classification_report
print(classification_report(Y_test_ndb,pred))


#deployment
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = flask(_name_)
model1 = pickle.load(open("model.pk1", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["PLOT"])
def predict():
    
    For rendering results on HTML GUI 
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text='Claims values are fraud or genuine $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    For rendering result on HTML GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Employee salary should be ${}' .format(output))
    