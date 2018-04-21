
# coding: utf-8

# In[395]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import linear_model, cross_validation
from sklearn.feature_selection import RFE
from sklearn import preprocessing

logistic_regression = linear_model.LogisticRegression()


# ### Load data files 

# In[396]:


waves = pd.read_csv('../multiclass/Wavelength.csv', header=None)
x = pd.read_csv('../multiclass/X.csv', header=None)
y = pd.read_csv('../multiclass/y.csv', header=None)


# ### Setting key colour values

# In[397]:


def decode_colours(toDecode):
    keys = {0: 'Blue', 1: 'Green', 2: 'Pink', 3: 'Red', 4: 'Yellow'}

    res = []
    for i in toDecode:
        res.append(keys[i])
        
    return res


# ### Set up plot parameters 

# In[398]:


def plot_parameters(x, y):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = x
    fig_size[1] = y
    plt.rcParams["figure.figsize"] = fig_size


# In[399]:


print(waves.size, x.size, y.size)


# ### Drop records with NaN values if any exists

# In[400]:


# Drop all rows that have NaN values
def drop_nan(waves, x, y):
    waves.dropna()
    x.dropna()
    y.dropna()
    
drop_nan(waves, x, y)


# ### Split data into training and testing sets 

# In[401]:


flat_waves = waves.T.as_matrix(columns=None).flatten()
x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.3, random_state=42)

print(x_train.size, y_train.size)
print(x_test.size, y_test.size)


# In[402]:


x_train.describe()


# ### Plot features for data visualization

# In[405]:


def plot_training_data(x_train, y_train, waves):
    index = 0
    y_train = y_train.as_matrix(columns=None)
    
    for ind, row in x_train.iterrows():
        if y_train[index] == 0:
            plt.plot(x_train.columns.values, row, color="blue")
        elif y_train[index] == 1:
            plt.plot(x_train.columns.values, row, color="lightgreen")
        elif y_train[index] == 2:
            plt.plot(x_train.columns.values, row, color="hotpink")
        elif y_train[index] == 3:
            plt.plot(x_train.columns.values, row, color="indianred")
        elif y_train[index] == 4:
            plt.plot(x_train.columns.values, row, color="yellow")    
        index += 1
    
    plt.xlabel("Feature index", fontdict=None, labelpad=None, fontsize=12)
    plt.ylabel("Optical reflectance intensity", fontdict=None, labelpad=None, fontsize=12)
    plt.yticks([-150, -100, -50, 0, 50, 100, 150], fontsize=12)  
    plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900], fontsize=12)  
    plt.xlim(xmin=0, xmax=921)

    plt.show()


# In[406]:


plot_training_data(x_train, y_train, flat_waves)


# In[409]:


def plot_histogram(feature_index, x_train, y_train):
    plot_parameters(5, 5)
    blue = []
    green = []
    pink = []
    red = []
    yellow = []

    y_train = y_train.as_matrix(columns=None)

    index = 0
    for i in x_train[feature_index]:
        if y_train[index] == 0:
            blue.append(i)
        elif y_train[index] == 1:
            green.append(i)
        elif y_train[index] == 2:
            pink.append(i)
        elif y_train[index] == 3:
            red.append(i)
        elif y_train[index] == 4:
            yellow.append(i)
        index += 1

    plt.xlabel("Intencity values", fontdict=None, labelpad=None, fontsize=12)
    plt.ylabel("Number of intencity values", fontdict=None, labelpad=None, fontsize=12)
    # plt.hist([blue, green, pink, red], 100, alpha=1, color=["blue", "lightgreen", "pink", "red"])

    plt.hist([pink], 5, alpha=0.8, color=["pink"])
    plt.hist([blue], 5, alpha=0.8, color=["blue"])
    plt.hist([green], 5, alpha=0.8, color=["lightgreen"])
    plt.hist([red], 5, alpha=0.8, color=["indianred"])
    plt.hist([yellow], 5, alpha=0.8, color=["yellow"])

    plt.show()


# In[410]:


plot_histogram(400, x_train, y_train)
plot_histogram(900, x_train, y_train)


# ### One feature experiment
# It trains on a single feature using logistic regression model and plots accuracy. In return, this indicates which single features have
# best accuracy and thus should be chosen. 

# In[414]:


def one_feature_experiment(x_train, y_train):
    accuracies = []
    
    for i in x_train:
        x_one = x_train[i].values.reshape(-1, 1)
        logistic_regression.fit(x_one, y_train)
        y_res = logistic_regression.predict(x_one)
        accuracies.append(accuracy_score(y_train, y_res))

    return accuracies


# In[415]:


def plot_one_feature_exp_results(acc):
    plot_parameters(30, 10)
    plt.xlabel("Feature index", fontdict=None, labelpad=None, fontsize=12)
    plt.ylabel("Accuracy", fontdict=None, labelpad=None, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900], fontsize=12)  
    plt.xlim(xmin=0, xmax=921)
    
    index = 0
    for i in acc:
        if i > 0.9:
            plt.axvspan(index, index, color='pink', alpha=0.8)
        elif i > 0.8:
            plt.axvspan(index, index, color='pink', alpha=0.6)
        elif i > 0.7:
            plt.axvspan(index, index, color='pink', alpha=0.4)
        elif i > 0.6:
            plt.axvspan(index, index, color='pink', alpha=0.2)
        elif i > 0.5:
            plt.axvspan(index, index, color='pink', alpha=0.1)
        index += 1
        
    plt.plot(acc)


# In[416]:


acc = one_feature_experiment(x_train, y_train)
plot_one_feature_exp_results(acc)


# ### RFE experiment on training data and experiment visualization

# In[419]:


def experiment(no_features):
    results = []
    
    for i in no_features:
        logistic = RFE(logistic_regression, i, step=1)
        logistic = logistic.fit(x_train, y_train)
        
        print(get_true_indices(logistic.support_))
        
        y_res = logistic.predict(x_train)
        accuracy = accuracy_score(y_train, y_res.ravel())
        
        print(accuracy)
        
        results.append(accuracy)
        
    return results


# In[420]:


scores = experiment([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(scores)


# In[426]:


def plot_accuracies_REF(scores):
    no_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plot_parameters(5, 5)
    plt.xlabel("Number of Features", fontdict=None, labelpad=None, fontsize=12)
    plt.ylabel("Accuracy", fontdict=None, labelpad=None, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(no_features, fontsize=12)
    plt.yticks(fontsize=12)
    plt.axvspan(5, 5, color='pink', alpha=1)
    plt.plot(no_features, scores, color="royalblue")


# In[427]:


plot_accuracies_REF(scores)


# ### Confusion matrix plotting function implementation

# In[428]:


def plot_confusion_matrix(cm, title='Confusion matrix - Training set', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1, 2, 3, 4], decode_colours([0, 1, 2, 3, 4]))
    plt.yticks([0, 1, 2, 3, 4], decode_colours([0, 1, 2, 3, 4]))
    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')


# ### Selects only the three best features from training dataframe 
# 
# Also, update the x_train and x_test with the new selected sets.

# In[430]:


def select_columns(x, selected):
    return x[selected]

x_train = select_columns(x_train, [421, 429, 250])
x_test = select_columns(x_test, [421, 429, 250])


# ### Run linear logistic regression model - training set using three features

# In[436]:


logistic_regression.fit(x_train, y_train)
y_res = logistic_regression.predict(x_train)

print(accuracy_score(y_train, y_res))
print(classification_report(y_train, y_res))


# In[440]:


plot_parameters(5, 5)
cm = confusion_matrix(y_train, y_res)
np.set_printoptions(precision=1)

# confusion matrix for training set with three features"
plot_confusion_matrix(cm)


# ### Linear support vector classifier - training set using three features

# In[443]:


from sklearn import svm

svm_model = svm.SVC(C = 1.0)
svm_model.fit(x_train, y_train)
svc_predicted = svm_model.predict(x_train)

print(accuracy_score(y_train, svc_predicted))
print(classification_report(y_train, svc_predicted))


# In[444]:


plot_parameters(5, 5)
cm = confusion_matrix(y_train, svc_predicted)
np.set_printoptions(precision=1)

# confusion matrix for training set with three features"
plot_confusion_matrix(cm)


# ### Testing SVC on testing set - with three features

# In[463]:


svm_model_final = svm_model.fit(x_train, y_train)
svc_predicted_test = svm_model_final.predict(x_test)

print(accuracy_score(y_test, svc_predicted_test))
print(classification_report(y_test, svc_predicted_test))


# ### Write to file function

# In[464]:


def write_to_file(df, path):
    df.to_csv(path, header=False, index=False)


# ### Final results to a file 'PredictedClasses.csv' 

# In[465]:


to_clasify = pd.read_csv('../multiclass/XToClassify.csv', header=None)

# get only the three features
X = select_columns(to_clasify, [421, 429, 250])
y_final_res = svm_model_final.predict(X)

print(y_final_res)
write_to_file(pd.DataFrame({ 'predicted': y_final_res} ), "../multiClassTask/PredictedClasses.csv")

