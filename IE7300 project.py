#!/usr/bin/env python
# coding: utf-8

# # Project Name: Default of credit cards client
# Author: Luyao Xu

# # Project Background:
# 
# Credit cards plays an important role in bank’s financial products. Nowadays, with the development of the science and technology, credit cards becoming one of a very important payment methods and becoming more and more popular for not only face-to face transaction, but also for online purchase.In 2005, There are 49 banks has credit card services in Taiwan, issued number of credit cards reached a very high number of 45 million. People with different ages, different education level, different gender with different usage becoming credit card holders. Therefore, prediction of the default of all clients in different conditions and investment of default issue is meaningful for banks and financial institution.
# 

# # problem statement：
# In this project, my purpose is to find the reasons that influencing the defaults of credit card clients and try to determine how much each factor will contribute to default of credit cards and build a model for predicting the default of credit card clients.
# 
# I will sample the dataset into train and test data for model training. In this project, I am going to build 3 models to make prediction: Random Forest, KNN and Logistic regression.  KNN is a type of classification where the function is only approximated locally and all computation is deferred until function evaluation.The random forest is a classification algorithm consisting of many decisions trees. Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In this coding process, I will mainly use the package of ‘sklearn’.By the end of this project, I will compare the performance of each model by comparing the prediction accuracy of each model on testing data , pick best model for future use.
# 
# 

# # Dataset Information : 
# This dataset aimed at the case of customers default payments in Taiwan and compares the predictive accuracy of probability of default of couple data mining methods. The dataset is one .xml form with 30,000 instances and 23 variables.
# 
# Dataset link:https://archive-beta.ics.uci.edu/ml/datasets/default+of+credit+card+clients 
# 

# # Data dictionary：
# This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
# X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
# X2: Gender (1 = male; 2 = female).
# X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
# X4: Marital status (1 = married; 2 = single; 3 = others).
# X5: Age (year).
# X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
# X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. 
# X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.
# 

# # Explore data:
#  I will explore the data by renaming and combining some columns to make them clearer for each variables and graph some charts to show correlations between variables, how they will affect the default of credit cards. 

# # Initialize Data

# In[290]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)

import warnings
warnings.filterwarnings('ignore')

from sklearn.base import clone
from sklearn.datasets import load_wine,load_boston
from sklearn.model_selection import StratifiedKFold,cross_validate
from sklearn.metrics import accuracy_score,precision_score,recall_score,mean_absolute_error,mean_squared_error,r2_score,make_scorer
from abc import ABC,abstractmethod
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[291]:


df = pd.read_csv('~/Desktop/IE7300/credit_card_default.csv',skiprows=1) 
df.head()


# In[292]:


# make a copy of the dataset for EDA
df1 = df.copy()
df1.describe().T


# In[293]:


#check dataset dimensions
print("Dimensions of original data:",df1.shape)


# In[294]:


# check null values
df1=df1.rename(columns={'default payment next month':'default_pay'})
print("Number of null values:", df1.isnull().sum())


# Since there is no na value in this dataset, so no need to deal with null values.

# # Data Visulization

# In[295]:


corrdata=df1[['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','default_pay']]
corr = corrdata.corr()
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True, annot_kws={"size": 15})


# From this correlated chart, we can see that pay_0 to pay_6 have a hihger correlation to each other, and limit_bal,sex,education,marriage and age are less correlated to each other. From this we can assume that people who pays the bill on time has a big possibility to oay the next bill on time.

# In[296]:


df1['default_pay'].value_counts()
plt.title('default payment of next month')
ax1=sns.countplot(x='default_pay',data=df1)
ax1.set_xticklabels(['No default','default'])
plt.show()


# From above graph,we can see that the original dataset is somehow imbalance.

# In[297]:


sns.pairplot(df1, hue = 'default_pay', vars = ['AGE', 'MARRIAGE', 'SEX', 'EDUCATION', 'LIMIT_BAL'] )
plt.show()


# Let's see how different preditors affect our target.

# In[298]:


#Education
df1['EDUCATION'].unique()


# In[299]:


ax2=sns.countplot(x='EDUCATION',hue='default_pay',data=df1)
ax2.set_xticklabels([' ','graduate school','university','high school','others',' ',' '],rotation=90)
plt.show()


# From this histagram we can see that people with a higher level education has more possibility to default than others.

# In[300]:


#age
df1['AGE'].unique()


# In[301]:


sns.displot(data=df1, x='AGE', hue='default_pay', fill=True, palette=sns.color_palette('bright')[:2],height=5, aspect=1.5)


# From age distribution we can see that people from age 20-30 is increasing to default,and start decreasing after age of 30, drop at significantly after age 40.

# In[302]:


#sex
ax3=sns.countplot(x='SEX',hue='default_pay',data=df1)
ax3.set_xticklabels(['Male','Female'],rotation=90)
plt.show()


# From female sex distribution we can see female has more default than male.

# In[303]:


plt.subplots(figsize=(20,10))

plt.subplot(231)
plt.scatter(x=df1.PAY_AMT1, y=df1.BILL_AMT1, c='r', s=1)

plt.subplot(232)
plt.scatter(x=df1.PAY_AMT2, y=df1.BILL_AMT2, c='b', s=1)

plt.subplot(233)
plt.scatter(x=df1.PAY_AMT3, y=df1.BILL_AMT3, c='g', s=1)

plt.subplot(234)
plt.scatter(x=df1.PAY_AMT4, y=df1.BILL_AMT4, c='c', s=1)
plt.ylabel("Bill Amount in past 6 months", fontsize=25)

plt.subplot(235)
plt.scatter(x=df1.PAY_AMT5, y=df1.BILL_AMT5, c='y', s=1)
plt.xlabel("Payment in past 6 months", fontsize=25)

plt.subplot(236)
plt.scatter(x=df1.PAY_AMT6, y=df1.BILL_AMT6, c='m', s=1)

plt.show()


# The plot indicates that clients who have a higher bill amount have a higher posibility to make a low payment.
# This we can infer since maximum number of datapoints are closely packed along the Y-axis near to 0 on X-axis

# # Train Test Split

# In[304]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X=df1.drop("default_pay",axis=1)
y=df1["default_pay"]
sc=StandardScaler()
X=sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)


# # Logistic regression

# In[305]:


class LogisticRegression():
    """
    Class for logisttic regression
    """

    def __init__(self, lr=0.001, epochs=1000):
        """
        Logistic Regression Constructor

        Args:
            lr (float, optional): _description_. Defaults to 0.001.
            epochs (int, optional): _description_. Defaults to 1000.
        """
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        
    def loss_function(self,x):
        """
        Sigmoid loss function

        Args:
            x (_type_): Z value(mx+b)

        Returns:
            _type_: Probability
        """
        return 1/(1+np.exp(-x))
    def fit(self, X, y):
        """
        Train the model

        Args:
            X (_type_): Features
            y (_type_): Response variable
        """
        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0

        for _ in range(self.epochs):
            linear_pred = np.dot(X, self.w) + self.b
            pred = self.loss_function(linear_pred)

            dw = (1/n) * np.dot(X.T, (pred - y))
            db = (1/n) * np.sum(pred-y)

            self.w = self.w - self.lr*dw
            self.b = self.b - self.lr*db

    def predict(self, X):
        """
        Predict the Y

        Args:
            X (_type_): _description_

        Returns:
            _type_: Y-hat probability
        """
        linear_pred = np.dot(X, self.w) + self.b
        y_pred = self.loss_function(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred


# In[306]:


lrModel = LogisticRegression()
lrModel.fit(X_train,y_train)


# In[307]:


y_predicted = lrModel.predict(X_test)
print('Accuracy Score: ', accuracy_score(y_test, y_predicted),'\n\n')    
print(classification_report(y_test, y_predicted))


# In[308]:


from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
metrics.confusion_matrix(y_test, y_predicted)


# The logistic model accuracy score is 0.80.23, which means the percentage of correct predictors is 80.23%. It seems not too bad with the accuracy we found so far. However, from the confusion matrix report and report details, even though the prediction correctness is okay, we can still found out that we only have a good prediction on our 1st class, but did a pretty bad prediction on our 2nd class, since both recall score and f1 score are pretty low.

# # PCA

# In[309]:


# PCA model custom
class PCA:
    """
     Implement the PCA from scratch
    """
    def __init__(self, n_components):
        """
         Constructor for PCA class

        Args:
            n_components (_type_): _description_
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fit the PCA model

        Args:
            X (_type_): _description_
        """
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, function needs samples as columns
        cov = np.cov(X.T)

        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors
        self.components = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)


# In[310]:


#Show top 2 of the PCA component output
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
print(x1)
print(x2)


# In[311]:


#Plot the PCA 1, PCA 2 , and PCA2 components
plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()


# From the above graph, we can see that some variations were captured by the principal components since there is some structure in the points when projected along the two principal component axis. The points belonging to the same class are close to each other, and the points or images that are very different semantically are further away from each other.So PCA model is not a choice for our further prediction from this case.

# # Random Forest

# In[312]:


#base class for the random forest algorithm
class RandomForest(ABC):
    #initializer
    def __init__(self,n_trees=100):
        self.n_trees = n_trees
        self.trees   = []
        
    #private function to make bootstrap samples
    def __make_bootstraps(self,data):
        #initialize output dictionary & unique value count
        dc   = {}
        unip = 0
        #get sample size
        b_size = data.shape[0]
        #get list of row indexes
        idx = [i for i in range(b_size)]
        #loop through the required number of bootstraps
        for b in range(self.n_trees):
            #obtain boostrap samples with replacement
            sidx   = np.random.choice(idx,replace=True,size=b_size)
            b_samp = data[sidx,:]
            #compute number of unique values contained in the bootstrap sample
            unip  += len(set(sidx))
            #obtain out-of-bag samples for the current b
            oidx   = list(set(idx) - set(sidx))
            o_samp = np.array([])
            if oidx:
                o_samp = data[oidx,:]
            #store results
            dc['boot_'+str(b)] = {'boot':b_samp,'test':o_samp}
        #return the bootstrap results
        return(dc)
  
    #public function to return model parameters
    def get_params(self, deep = False):
        return {'n_trees':self.n_trees}

    #protected function to obtain the right decision tree
    @abstractmethod
    def _make_tree_model(self):
        pass
    #protected function to train the ensemble
    def _train(self,X_train,y_train):
        #package the input data
        training_data = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
        #make bootstrap samples
        dcBoot = self.__make_bootstraps(training_data)
        #iterate through each bootstrap sample & fit a model ##
        tree_m = self._make_tree_model()
        dcOob    = {}
        for b in dcBoot:
            #make a clone of the model
            model = clone(tree_m)
            #fit a decision tree model to the current sample
            model.fit(dcBoot[b]['boot'][:,:-1],dcBoot[b]['boot'][:,-1].reshape(-1, 1))
            #append the fitted model
            self.trees.append(model)
            #store the out-of-bag test set for the current bootstrap
            if dcBoot[b]['test'].size:
                dcOob[b] = dcBoot[b]['test']
            else:
                dcOob[b] = np.array([])
        #return the oob data set
        return(dcOob)
    #protected function to predict from the ensemble
    def _predict(self,X):
        #check we've fit the ensemble
        if not self.trees:
            print('You must train the ensemble before making predictions!')
            return(None)
        #loop through each fitted model
        predictions = []
        for m in self.trees:
            #make predictions on the input X
            yp = m.predict(X)
            #append predictions to storage list
            predictions.append(yp.reshape(-1,1))
        #compute the ensemble prediction
        ypred = np.mean(np.concatenate(predictions,axis=1),axis=1)
        #return the prediction
        return(ypred)
     


# In[313]:


#class to control tree node
class Node:
    #initializer
    def __init__(self):
        self.__Bs    = None
        self.__Bf    = None
        self.__left  = None
        self.__right = None
        self.leafv   = None

    #set the split,feature parameters for this node
    def set_params(self,Bs,Bf):
        self.__Bs = Bs
        self.__Bf = Bf
        
    #get the split,feature parameters for this node
    def get_params(self):
        return(self.__Bs,self.__Bf)    
        
    #set the left/right children nodes for this current node
    def set_children(self,left,right):
        self.__left  = left
        self.__right = right
        
    #get the left child node
    def get_left_node(self):
        return(self.__left)
    
    #get the right child node
    def get_right_node(self):
        return(self.__right)


# In[314]:


#class for random forest classifier
class RandomForestClassifier(RandomForest):
    #initializer
    def __init__(self,n_trees=100,max_depth=None,min_samples_split=2,loss='gini',balance_class_weights=False):
        super().__init__(n_trees)
        self.max_depth             = max_depth
        self.min_samples_split     = min_samples_split
        self.loss                  = loss
        self.balance_class_weights = balance_class_weights
        
    #protected function to obtain the right decision tree
    def _make_tree_model(self):
        return(DecisionTreeClassifier())
    
    #public function to return model parameters
    def get_params(self, deep = False):
        return {'n_trees':self.n_trees,
                'max_depth':self.max_depth,
                'min_samples_split':self.min_samples_split,
                'loss':self.loss,
                'balance_class_weights':self.balance_class_weights}
    
    #train the ensemble
    def fit(self,X_train,y_train,print_metrics=False):
        #call the protected training method
        dcOob = self._train(X_train,y_train)
        #if selected, compute the standard errors and print them
        if print_metrics:
            #initialise metric arrays
            accs = np.array([])
            pres = np.array([])
            recs = np.array([])
            #loop through each bootstrap sample
            for b,m in zip(dcOob,self.trees):
                #compute the predictions on the out-of-bag test set & compute metrics
                if dcOob[b].size:
                    yp  = m.predict(dcOob[b][:,:-1])
                    acc = accuracy_score(dcOob[b][:,-1],yp)
                    pre = precision_score(dcOob[b][:,-1],yp,average='weighted')   
                    rec = recall_score(dcOob[b][:,-1],yp,average='weighted')
                    #store the error metrics
                    accs = np.concatenate((accs,acc.flatten()))
                    pres = np.concatenate((pres,pre.flatten()))
                    recs = np.concatenate((recs,rec.flatten()))
            #print standard errors
            print("Standard error in accuracy: %.2f" % np.std(accs))
            print("Standard error in precision: %.2f" % np.std(pres))
            print("Standard error in recall: %.2f" % np.std(recs))
            
    #predict from the ensemble
    def predict(self,X):
        #call the protected prediction method
        ypred = self._predict(X)
        #convert the results into integer values & return
        return(np.round(ypred).astype(int))
    


# In[315]:


rfc = RandomForestClassifier(balance_class_weights=True)
rfc.fit(X_train,y_train.values)


# In[316]:


y_predicted = rfc.predict(X_test)
print('Accuracy Score: ', accuracy_score(y_test, y_predicted),'\n\n')    
print(classification_report(y_test, y_predicted))


# In[317]:


metrics.confusion_matrix(y_test, y_predicted)


# Our Random Forest model has an accuracy of 0.818,which means the percentage of correct predictors is 81.8%.As we can see, not only the accuracy and confusion matrix performs better,the precision and f1 score also increased, thus RF model performs a little better than our logistic model.

# # KNN

# In[318]:


from scipy import stats
from typing import Dict, Any
from abc import ABC,abstractmethod
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import mean_squared_error,                            mean_absolute_error,                            accuracy_score,                            precision_score,                            recall_score,                            f1_score,                            make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
class KNN(ABC):
    """
    Base class for KNN implementations
    """
    
    def __init__(self, K : int = 3, metric : str = 'minkowski', p : int = 2) -> None:
        """
        Initializer function. Ensure that input parameters are compatiable.
        Inputs:
            K      -> integer specifying number of neighbours to consider
            metric -> string to indicate the distance metric to use (valid entries are 'minkowski' or 'cosine')
            p      -> order of the minkowski metric (valid only when distance == 'minkowski')
        """
        # check distance is a valid entry
        valid_distance = ['minkowski','cosine']
        if metric not in valid_distance:
            msg = "Entered value for metric is not valid. Pick one of {}".format(valid_distance)
            raise ValueError(msg)
        # check minkowski p parameter
        if (metric == 'minkowski') and (p <= 0):
            msg = "Entered value for p is not valid. For metric = 'minkowski', p >= 1"
            raise ValueError(msg)
        # store/initialise input parameters
        self.K       = K
        self.metric  = metric
        self.p       = p
        self.X_train = np.array([])
        self.y_train = np.array([])
    def __del__(self) -> None:
        """
        Destructor function. 
        """
        del self.K
        del self.metric
        del self.p
        del self.X_train
        del self.y_train
      
    def __minkowski(self, x : np.array) -> np.array:
        """
        Private function to compute the minkowski distance between point x and the training data X
        Inputs:
            x -> numpy data point of predictors to consider
        Outputs:
            np.array -> numpy array of the computed distances
        """
        return np.power(np.sum(np.power(np.abs(self.X_train - x),self.p),axis=1),1/self.p)
    
    def __cosine(self, x : np.array) -> np.array:
        """
        Private function to compute the cosine distance between point x and the training data X
        Inputs:
            x -> numpy data point of predictors to consider
        Outputs:
            np.array -> numpy array of the computed distances
        """
        return (1 - (np.dot(self.X_train,x)/(np.linalg.norm(x)*np.linalg.norm(self.X_train,axis=1))))
    
    def __distances(self, X : np.array) -> np.array:
        """
        Private function to compute distances to each point x in X[x,:]
        Inputs:
            X -> numpy array of points [x]
        Outputs:
            D -> numpy array containing distances from x to all points in the training set.
        """
        # cover distance calculation
        if self.metric == 'minkowski':
            D = np.apply_along_axis(self.__minkowski,1,X)
        elif self.metric == 'cosine':
            D = np.apply_along_axis(self.__cosine,1,X)
        # return computed distances
        return D
    
    @abstractmethod
    def _generate_predictions(self, idx_neighbours : np.array) -> np.array:
        """
        Protected function to compute predictions from the K nearest neighbours
        """
        pass
        
    def fit(self, X : np.array, y : np.array) -> None:
        """
        Public training function for the class. It is assummed input X has been normalised.
        Inputs:
            X -> numpy array containing the predictor features
            y -> numpy array containing the labels associated with each value in X
        """
        # store training data
        self.X_train = np.copy(X)
        self.y_train = np.copy(y)
    
    def predict(self, X : np.array) -> np.array:
        """
        Public prediction function for the class. 
        It is assummed input X has been normalised in the same fashion as the input to the training function
        Inputs:
            X -> numpy array containing the predictor features
        Outputs:
           y_pred -> numpy array containing the predicted labels
        """
        # ensure we have already trained the instance
        if (self.X_train.size == 0) or (self.y_train.size == 0):
            raise Exception('Model is not trained. Call fit before calling predict.')
        # compute distances
        D = self.__distances(X)
        # obtain indices for the K nearest neighbours
        idx_neighbours = D.argsort()[:,:self.K]
        # compute predictions
        y_pred = self._generate_predictions(idx_neighbours)
        # return results
        return y_pred
    
    def get_params(self, deep : bool = False) -> Dict:
        """
        Public function to return model parameters
        Inputs:
            deep -> boolean input parameter
        Outputs:
            Dict -> dictionary of stored class input parameters
        """
        return {'K':self.K,
                'metric':self.metric,
                'p':self.p}
    


# In[319]:


class KNNClassifier(KNN):
    """
    Class for KNN classifiction implementation
    """
    
    def __init__(self, K : int = 3, metric : str = 'minkowski', p : int = 2) -> None:
        """
        Initializer function. Ensure that input parameters are compatiable.
        Inputs:
            K       -> integer specifying number of neighbours to consider
            metric  -> string to indicate the distance metric to use (valid entries are 'minkowski' or 'cosine')
            p       -> order of the minkowski metric (valid only when distance == 'minkowski')
        """
        # call base class initialiser
        super().__init__(K,metric,p)
        
    def _generate_predictions(self, idx_neighbours : np.array) -> np.array:
        """
        Protected function to compute predictions from the K nearest neighbours
        Inputs:
            idx_neighbours -> indices of nearest neighbours
        Outputs:
            y_pred -> numpy array of prediction results
        """        
        # compute the mode label for each submitted sample
        y_pred = stats.mode(self.y_train[idx_neighbours],axis=1).mode.flatten()   
        # return result
        return y_pred


# In[320]:


## define a helper function for our analysis ##
scoring_metrics = {'accuracy' : make_scorer(accuracy_score), 
                   'precision': make_scorer(precision_score),
                   'recall'   : make_scorer(recall_score),
                   'f1'       : make_scorer(f1_score)}
def cv_classifier_analysis(pipe : Any, 
                           X : np.array, 
                           y : np.array, 
                           k : int, 
                           scoring_metrics : Dict,
                           metric : str) -> None:
    """
    Function to carry out cross-validation analysis for input KNN classifier
    Inputs:
        pipe            -> input pipeline containing preprocessing and KNN classifier
        X               -> numpy array of predictors
        y               -> numpy array of labels
        k               -> integer value for number of nearest neighbours to consider
        scoring_metrics -> dictionary of scoring metrics to consider 
        metric          -> string indicating distance metric used
    """
    # print hyperparameter configuration
    print('RESULTS FOR K = {0}, {1}'.format(k,metric))
    # run cross validation
    dcScores = cross_validate(pipe,X,y,cv=StratifiedKFold(10),scoring=scoring_metrics)
    # report results
    print('Mean Accuracy: %.2f' % np.mean(dcScores['test_accuracy']))
    print('Mean Precision: %.2f' % np.mean(dcScores['test_precision']))
    print('Mean Recall: %.2f' % np.mean(dcScores['test_recall']))
    print('Mean F1: %.2f' % np.mean(dcScores['test_f1']))


# In[321]:


## perform cross-validation for a range of model hyperparameters for the Custom model ##
K = [3,6,9]
for k in K:
    # define the pipeline for manhatten distance
    p_manhat = Pipeline([('scaler', StandardScaler()), ('knn', KNNClassifier(k, metric = 'minkowski', p = 1))])
    
    cv_classifier_analysis(p_manhat, X, y, k, scoring_metrics, 'MANHATTEN DISTANCE')
    # cross validate for p_euclid


# Here I used 10-fold cross-validation to measure the performance of the KNN classifier. I have tried the values of k=3,6,9 in this model with a manhatten distance to compare the results. With our KNN performance, we got an accuracy of 0.77(the percentage of correct predictors is 77%) with k=3, ccuracy of 0.80(the percentage of correct predictors is 80%) with k=6 and ccuracy of 0.80(the percentage of correct predictors is 80%) with k=9. We can see that our accuracy resut of k values of 6 and 9 are the same, but when k=9 we have a little bit higher value in our recall and f1 score,which is the best prediction among all these k values.However, our random forest model still has a best performance with the accuracy of 81.8% so far. 

# # SVM

# In[322]:


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0,-1, 1)
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


# In[323]:


clf = SVM()
X_train1=np.array(X_train)
clf.fit(X_train1, y_train)
#print("Accuracy of SVM is ",round(clf.score(X_test, y_test),4))


# In[324]:


y_predicted = clf.predict(X_test)
y_predicted[y_predicted ==-1 ] = 0
print('Accuracy Score: ', accuracy_score(y_test, y_predicted),'\n\n')    
print(classification_report(y_test, y_predicted))


# Our SVM model has an accuracy of 0.8102, which means the percentage of correct predictors is 81.02%, here we can see our SVM model has an similiar accuracy with our random forest model of a lightly higer accuracy of 81.8%. Since our dataset is imblanced, our accuracy prediction could be affected due to the imblance dataset, so I will apply a up/under sampling method below to resample our dataset and compare these 2 models again.

# # Up/Under sampling-Random Forest

# In[346]:


from sklearn.utils import resample
X=df1.drop("default_pay",axis=1)
y=df1["default_pay"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

training_set = pd.concat([X_train, y_train], axis=1)
default = training_set[training_set.default_pay == 1]
not_default = training_set[training_set.default_pay == 0]


# In[347]:


undersample = resample(not_default, 
                       replace=True, 
                       n_samples=len(default), #set the number of samples to equal the number of the majority class
                       random_state=42)
# Returning to new training set
undersample_train = pd.concat([default, undersample])
undersample_train.default_pay.value_counts(normalize=True)
undersample_x_train = undersample_train.drop('default_pay', axis=1)
undersample_y_train = undersample_train.default_pay


# In[348]:


undersample_train.default_pay.value_counts()


# In[349]:


oversample = resample(default, 
                       replace=True, 
                       n_samples=len(not_default), #set the number of samples to equal the number of the majority class
                       random_state=42)
# Returning to new training set
oversample_train = pd.concat([not_default, oversample])
oversample_train.default_pay.value_counts(normalize=True)
oversample_x_train = oversample_train.drop('default_pay', axis=1)
oversample_y_train = oversample_train.default_pay


# In[350]:


oversample_train.default_pay.value_counts()


# In[351]:


undersample_rfc=rfc.fit(undersample_x_train, undersample_y_train.values)
y_predicted = rfc.predict(X_test)
print('Rfc undersample Accuracy Score: ', accuracy_score(y_test, y_predicted),'\n\n')
    
print(classification_report(y_test, y_predicted))


# In[352]:


oversample_rfc=rfc.fit(oversample_x_train, oversample_y_train.values)
y_predicted = rfc.predict(X_test)
print('Rfc oversample Accuracy Score: ', accuracy_score(y_test, y_predicted),'\n\n')
    
print(classification_report(y_test, y_predicted))


# # up/under sample-SVM

# In[362]:


undersample_SVM=clf.fit(np.array(undersample_x_train), undersample_y_train)


# In[365]:


y_predicted = clf.predict(X_test)
y_predicted[y_predicted ==-1 ] = 0
print('SVM undersample Accuracy Score: ', accuracy_score(y_test, y_predicted),'\n\n')
    
print(classification_report(y_test, y_predicted))


# In[366]:


oversample_SVM=clf.fit(np.array(oversample_x_train), oversample_y_train)


# In[367]:


y_predicted = clf.predict(X_test)
y_predicted[y_predicted ==-1 ] = 1
print('SVM oversample Accuracy Score: ', accuracy_score(y_test, y_predicted),'\n\n')
    
print(classification_report(y_test, y_predicted))


# # Model performance

# We have conducted various machine learning models on the dataset, also trained several classifiers can be used for predictive analysis. The next mission is to pick top ones to be used under business environment by comparing the prediction accuracy of each model. We have compared the prediction accuracy of each model on testing dataset. As we can see, by using original training set, the accuracy of Logistic, RF ,KNN and SVM are all around 80% 
# 
# However, detecting potential late payments for the credit account is a special task that we should focus more on how successfully the model can predict to recognize clients with late payment potential. So the accuracy rate is not the only factor that we should consider in this case,we should also consider the score of precision, recall and f1 scores at the same time.
# after resampling, we can see there is a decrease on accuracy of 2 models, and not a very much change on other scores as well, so we can conclude that the imblance in this dataset not affect our prediction very much. 
# 
# Based on all information we have so far, our Logistic model, RFC and SVM are 3 models with similiar performance.So we can pick these 3 models to be used for further learning process to improve.

# # Conclusion

# Based on the some performance of my better models, I have concluded some suggestions for credit card clients and banks:
# 
# 1. Banks should put more attentions on the customers who is more educated,single, between 30-40 years old,as they will more likely to make payments on time everymonth.And they should remind those customers who are tend to be with late payment.
# 
# 2. Customers should check their bank accounts frequently to ensure the payment ontime.
