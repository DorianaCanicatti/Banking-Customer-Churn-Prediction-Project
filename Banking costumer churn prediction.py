# This is the python version of a project whose aim is to
# predict the possibility of abandonment of a bank by its costumers
# Dataset: Kaggle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.utils import column_or_1d

# First we read the excel file that contains our data
data = pd.read_csv("Banking_customer_churn.csv")
data =pd.DataFrame(data)

############ DATA EXPLORATION AND PRE-PROCESSING ############################
print(data.head()) #first 5 rows
print(data.info())
# 14 variables, each row is a costumer (total = 10.000) and we have different informations about them starting from its ID
# The variables that we want to consider the dependent one is whether they exited the bank or no (binary variable 0-1)
# from the command info we observe that it is considered an integer and not a boolean, we will modify it

print(data.describe())
# from the describe  function we can observe that the mean for the exited variable is 0.204, so we assume that
# on average the majority of our costumer didn't exited the bank (0.20 is closer to 0 than to 1)
# we can observe other informations such as the mean, the min/max of all the other variables but we focus on this one

print(data.dropna()) #no NA, we saw that in the "info" command
print(data.drop_duplicates()) #no duplicates, still 10.000 rows

######################### DATA GROUPING ##################################à
print(data.groupby("Exited")["Exited"].count())
# to find them singularly (we will need it in the future)
print(data[data["Exited"]==0]["Exited"].count())
print(data[data["Exited"]==1]["Exited"].count())
# we observe 7963 users who didn't exit and 2037 who did
# our dataset is unbalanced

############################# DATA BALANCING ###############################################
# We over-sample the minority class
minority_class = data[data["Exited"]==1]
majority_class = data[data["Exited"]==0]

# The new minority class will be the 90% of the majority one
minority_class_oversampled = minority_class.sample(frac=0.9*len(majority_class)/len(minority_class), replace=True)

# Combination of the classes and random sampling
data_balanced = pd.concat([majority_class, minority_class_oversampled])
newdata = data_balanced.sample(frac=1).reset_index(drop=True)
print(newdata['Exited'].value_counts())
# now we have a more balanced sample
# 7963 = No
# 7167 = Yes
N=len(majority_class)+len(minority_class_oversampled)

##################################### DATA VISUALIZATION #######################################à
# we have to transform the Exited variable in a boolean one
newdata['Exited'] = newdata['Exited'].astype(bool)
xval=["No","Yes"]

palette={"orange","purple"}
plt.figure(figsize=(10,6))
plt.bar(xval, newdata["Exited"].value_counts(), color=palette)
plt.title("Exited and non-exited clients")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.show()

# to see the more detailed plots about all the other variables conditioned by the "exited" one
# go check the R-file

################################# CORRELATION #########################################################à
# First we have to remove the non numeric variables with the exception of "gender"
newdata1=newdata.drop(columns=["RowNumber","CustomerId","Surname", "Geography"])
# Transforming the gender variable in a binary variable
newdata1['Gender'] = newdata1['Gender'].map({'Male': 0, 'Female': 1})
print(newdata1.info())

# Correlation
corr = newdata1.corr()
# Create the map
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, center=0, linewidths=0.5, linecolor='black')
plt.title('Correlation Heatmap')
plt.show()

# COMMENT:
# The correlation between the variables is almost null
# The most relevant relationships are between the number of products and the balance, and between age and Exited
# the higher the balance the lower the number of banking product
# The higher the age the bigger is the value of Exited, which means it is more probable it is equal to 1 = yes

################################### PRINCIPAL COMPONENT ANALYSIS ################################################à
# as we know, to compute the principal component variables we must have a
# significant correlation between the original variables, condition that we don't have
# we will try the same to see what we obtain

from sklearn.decomposition import PCA

pca = PCA(n_components=3).fit(corr) #we choose 3 components
print(pca.explained_variance_ratio_) #this is the proportion of variance explained by each component

# we try with more components
pca2 = PCA(n_components=6).fit(corr) #we choose 3 components
print(pca2.explained_variance_ratio_) #this is the proportion of variance explained by each component

# we need a lot of components to have a decent amount of variance explained, the pca in this way doesn't make sense
# pca works better with scaled data, but being this machine learning useless we skip this step

#######################################à LOGISTIC REGRESSION ###########################################
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random

# We divide the dataset in training and testing set
Y=pd.DataFrame(newdata1.loc[:,"Exited"])
Y=column_or_1d(Y, warn=True) # we transform the dependent variable in 1 column vector
X=pd.DataFrame(newdata1.iloc[:, 0:9])
print(X.info())
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
# 20% of the sample is used as test

# Scaling
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
# It's not necessary to scale the y variable

# Regression
log_reg_full= sm.Logit(y_train, X_train_scaled).fit()
print(log_reg_full.summary()) #we should remove variables x4 (tenure)- 6 (NumofProd)- 7 (hasCred)- 9 (EstimatedSal)
# The LL of the null is bigger than the LL of the model
# Create the model using the training sets

# Prediction of the y of the testing
y_test_pred_prob = log_reg_full.predict(X_test_scaled)
# Prediction of the y of the training
y_train_pred_prob = log_reg_full.predict(X_train_scaled)

# Conversion of probabilities in binary results (0 or 1) with a threshold of 0.5
y_test_pred = np.where(y_test_pred_prob >= 0.5, 1, 0)
y_train_pred = np.where(y_train_pred_prob >= 0.5, 1, 0)

# Calculating the accuracy and making the confusion matrix
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Accuracy of the full model on the training set:", round(train_accuracy*100,2),"%")
print("Accuracy of the full model on the testing set:", round(test_accuracy*100,2),"%")
print("Absence of overfitting.")
# THE ACCURACY FOR LOGISTIC REGRESSION IS ALMOST 70% IN BOTH CASES

##### RESTRICTED MODEL ########
# We try the same with a restricted model (see the R file and how we chose it by using LRTest)
# Restricted model: Exited ~ CreditScore + Gender + Age + Balance + IsActiveMember
X2=pd.DataFrame(X.iloc[:, [0,1,2,4,7]])
print(X2.info())
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y, test_size=0.25, random_state=42)
X2_train_scaled = scaler_X.fit_transform(X2_train)
X2_test_scaled = scaler_X.transform(X2_test)

# Regression with restricted
log_reg_rest = sm.Logit(y2_train, X2_train_scaled).fit()
print(log_reg_rest.summary()) # all the p-values of the variables are smaller than alpha 0.05
# the LL of the null is still bigger than the one of the model

# Prediction of the y of the testing
y_test_pred_prob2 = log_reg_rest.predict(X2_test_scaled)
# Prediction of the y of the training
y_train_pred_prob2 = log_reg_rest.predict(X2_train_scaled)

# Conversion of probabilities in binary results (0 or 1) with a threshold of 0.5
y_test_pred2 = np.where(y_test_pred_prob2 >= 0.5, 1, 0)
y_train_pred2 = np.where(y_train_pred_prob2 >= 0.5, 1, 0)

# Calculating the accuracy and making the confusion matrix
train_accuracy2 = accuracy_score(y_train, y_train_pred2)
test_accuracy2 = accuracy_score(y_test, y_test_pred2)

print("Accuracy of the restricted model on the training set:", round(train_accuracy2*100,2),"%")
print("Accuracy of the restricted model on the testing set:", round(test_accuracy2*100,2),"%")
print("Absence of overfitting.")
# THE ACCURACY FOR LOGISTIC REGRESSION IS ALMOST 70% IN BOTH CASES
# FOR A SMALL DIFFERENCE IN THE ACCURACY, THIS MODEL HAS MORE SIGNIFICANT VARIABLES
# THE RESULTS ARE THE SAME OBTAINED IN R

# LR TEST
LR_stat = 2 * (log_reg_full.llf - log_reg_rest.llf)
df_diff = log_reg_full.df_model - log_reg_rest.df_model  # Differenza dei gradi di libertà
p_value = sm.stats.chisquare_effectsize(LR_stat, df_diff)
print(f"\nLikelihood Ratio Test statistic: {LR_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print("P-value is smaller than 0.05, we don't reject the hypothesis that the restricted model is better.")

################################# LINEAR DISCRIMINANT ANALYSIS #########################################################

# To compute the LDA we must have normally distributed variables, we can check this through Mardia's test
# Mardia's test:
# H0 = our predictors are distributed as a multivariate normal
# H1 = our predictors are distributed as a multivariate normal

import pingouin as pg

mardia_test = pg.multivariate_normality(X2, alpha=0.05)
print(mardia_test)
print("The p-value is smaller that 0.05, so we don't accept the null hypothesis about the multivariate normality of the variables.")
print("It's not possible to compute the linear discriminant analysis.")

####################################### NEAREST NEIGHBOR ###############################################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# We will use the same X_train and X_test data used for the restricted model of logistic regression
# The model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X2_train_scaled, y_train)

# Prediction of the testing and the training
y_test_prob_knn = knn.predict(X2_test_scaled)
y_train_prob_knn = knn.predict(X2_train_scaled)

# Conversion
y_test_pred_knn = np.where(y_test_prob_knn >= 0.5, 1, 0)
y_train_pred_knn = np.where(y_train_prob_knn >= 0.5, 1, 0)

# Calculating the accuracies
knn_train_accuracy = accuracy_score(y_train, y_train_pred_knn)
print("KNN model accuracy  on the training set:", round(knn_train_accuracy*100,2),"%")
knn_test_accuracy = accuracy_score(y_test, y_test_pred_knn)
print("KNN model accuracy on the testing set:", round(knn_test_accuracy*100,2),"%")
print("In the R-file we obtained different accuracies and this led as to assume overfitting, a possible cause was that we had not scaled the variables."
      "In this case there is no overfitting probably because this transformation has been made.")

####################################### SUPPORT VECTOR MACHINE #########################################################
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error

# The model
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X2_train_scaled, y_train)

# Prediction
y_train_prob_svm = svm_classifier.predict(X2_train_scaled)
y_test_prob_svm = svm_classifier.predict(X2_test_scaled)

# Conversion
y_test_pred_svm = np.where(y_test_prob_svm >= 0.5, 1, 0)
y_train_pred_svm = np.where(y_train_prob_svm >= 0.5, 1, 0)

# Calculating the accuracies
svm_train_accuracy = accuracy_score(y_train, y_train_pred_svm)
print("SVM model accuracy  on the training set:", round(svm_train_accuracy*100,2),"%")
svm_test_accuracy = accuracy_score(y_test, y_test_pred_svm)
print("SVM model accuracy on the testing set:", round(svm_test_accuracy*100,2),"%")
print("The accuracy are both almost 70%, there is no overfitting.")

############################################## DECISION TREE ###########################################################
from sklearn.tree import DecisionTreeClassifier

#The model
dt = DecisionTreeClassifier(criterion='gini', random_state=42)
dt.fit(X2_train_scaled, y_train)

# Prediction
y_train_prob_dt = dt.predict(X2_train_scaled)
y_test_prob_dt = dt.predict(X2_test_scaled)

# Conversion
y_test_pred_dt = np.where(y_test_prob_dt >= 0.5, 1, 0)
y_train_pred_dt = np.where(y_train_prob_dt >= 0.5, 1, 0)

# Calculating the accuracies
dt_train_accuracy = accuracy_score(y_train, y_train_pred_dt)
print("Decision tree model accuracy  on the training set:", round(dt_train_accuracy*100,2),"%")
dt_test_accuracy = accuracy_score(y_test, y_test_pred_dt)
print("Decision tree model accuracy on the testing set:", round(dt_test_accuracy*100,2),"%")
print("The accuracy are a bit different, we have the highest values of accuracy with a bit of overfitting."
      "The values are different from the R-file, assuming that the scaling is the cause.")

################################################# RANDOM FOREST ########################################################

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
# n. estimators = n. of trees, if accuracy is low you choose a bigger value
# max depth = complexity of the trees, if there is overfitting you reduce it
rf.fit(X2_train_scaled, y_train)

# Prediction
y_train_prob_rf = rf.predict(X2_train_scaled)
y_test_prob_rf = rf.predict(X2_test_scaled)

# Conversion
y_test_pred_rf = np.where(y_test_prob_rf >= 0.5, 1, 0)
y_train_pred_rf = np.where(y_train_prob_rf >= 0.5, 1, 0)

# Calculating the accuracies
rf_train_accuracy = accuracy_score(y_train, y_train_pred_rf)
print("Random forest model accuracy  on the training set:", round(rf_train_accuracy*100,2),"%")
rf_test_accuracy = accuracy_score(y_test, y_test_pred_rf)
print("Random forest model accuracy on the testing set:", round(rf_test_accuracy*100,2),"%")
print("The accuracy are both almost 73%, there is no overfitting.")

################################################# NEURAL NETWORK #######################################################

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500, random_state=42)
mlp.fit(X2_train_scaled, y_train)

# Prediction
y_train_prob_nn = mlp.predict(X2_train_scaled)
y_test_prob_nn = mlp.predict(X2_test_scaled)

# Conversion
y_test_pred_nn = np.where(y_test_prob_nn >= 0.5, 1, 0)
y_train_pred_nn = np.where(y_train_prob_nn >= 0.5, 1, 0)

# Calculating the accuracies
nn_train_accuracy = accuracy_score(y_train, y_train_pred_nn)
print("Neural network model accuracy  on the training set:", round(nn_train_accuracy*100,2),"%")
nn_test_accuracy = accuracy_score(y_test, y_test_pred_nn)
print("Neural network model accuracy on the testing set:", round(nn_test_accuracy*100,2),"%")
print("The accuracy is very low, the model is not good as in the R-file.")

# ATTENTION:  the nodes and the hidden levels have been changed with respect to R to increase accuracy.