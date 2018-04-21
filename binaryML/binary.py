
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import linear_model
from sklearn.feature_selection import RFE

logistic_regression = linear_model.LogisticRegression()


# ### Load data files

# In[2]:


waves = pd.read_csv('../binary/Wavelength.csv', header=None)
x = pd.read_csv('../binary/X.csv', header=None)
y = pd.read_csv('../binary/y.csv', header=None)


# ### Set up plot parameters

# In[3]:


def plot_parameters(x, y):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = x
    fig_size[1] = y
    plt.rcParams["figure.figsize"] = fig_size

plot_parameters(30, 10)


# ### Drop records with NaN values if any exists

# In[4]:


# Drop all rows that have NaN values
def drop_nan(waves, x, y):
    waves.dropna()
    x.dropna()
    y.dropna()
    
drop_nan(waves, x, y)


# ### Split data into training and testing sets 

# In[5]:


flat_waves = waves.T.as_matrix(columns=None).flatten()
x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.3, random_state=42)

print(x_train.size, y_train.size)
print(x_test.size, y_test.size)


# In[6]:


y_train = y_train.as_matrix(columns=None)


# ### Inspect traning set features

# In[7]:


x_train.describe()


# ### Plot all features

# In[8]:


def plot_features(x_train, y_train):
    index = 0
    for ind, row in x_train.iterrows():
        if y_train[index] == 1:
            plt.plot(x_train.columns.values, row, color="indianred")
        else:
            plt.plot(x_train.columns.values, row, color="lightgreen")
        index += 1

    plt.xlabel("Feature index", fontdict=None, labelpad=None, fontsize=12)
    plt.ylabel("Optical reflectance intensity", fontdict=None, labelpad=None, fontsize=12)
    plt.yticks([-150, -100, -50, 0, 50, 100, 150], fontsize=12)  
    plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900], fontsize=12)  
    plt.xlim(xmin=0, xmax=921)

#     plt.axvspan(650, 650, color='grey', alpha=0.5)

    plt.show()


# In[9]:


plot_features(x_train, y_train)


# ### One feature experiment
# It trains on a single feature using logistic regression model and plots accuracy. In return, this indicates which single features have
# best accuracy and thus should be chosen. 

# In[10]:


def one_feature_experiment(x_train, y_train):
    accuracies = []
    
    for i in x_train:
        x_one = x_train[i].values.reshape(-1, 1)
        logistic_regression.fit(x_one, y_train.ravel())
        y_res = logistic_regression.predict(x_one)
        accuracies.append(accuracy_score(y_train, y_res))

    return accuracies


# In[11]:


def plot_one_feature_exp_results(acc):
    plot_parameters(30, 10)
    plt.xlabel("Feature index", fontdict=None, labelpad=None, fontsize=12)
    plt.ylabel("Accuracy", fontdict=None, labelpad=None, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900], fontsize=12)  
    plt.xlim(xmin=0, xmax=921)
    
    index = 0
    for i in acc:
        if i == 1:
            plt.axvspan(index, index+1, color='pink', alpha=0.8)
        elif i > 0.9:
            plt.axvspan(index, index, color='pink', alpha=0.6)
        elif i > 0.8:
            plt.axvspan(index, index, color='pink', alpha=0.4)
        elif i > 0.7:
            plt.axvspan(index, index, color='pink', alpha=0.2)
        elif i > 0.6:
            plt.axvspan(index, index, color='pink', alpha=0.1)
        elif i > 0.5:
            plt.axvspan(index, index, color='pink', alpha=0.05)

        index += 1
        
    plt.plot(acc)


# In[12]:


acc = one_feature_experiment(x_train, y_train)
plot_one_feature_exp_results(acc)


# ### Further feature investigation

# In[13]:


plot_parameters(10, 10)
green = []
red = []

index = 0
for i in x_train[620]:
    if y_train[index] == 1:
        red.append(i)
    else:
        green.append(i)
    index += 1
    
plt.xlabel("Intencity values", fontdict=None, labelpad=None, fontsize=12)
plt.ylabel("Number of intencity values", fontdict=None, labelpad=None, fontsize=12)
plt.hist([green, red], 20, alpha=1, color=["lightgreen", "indianred"])
plt.show()


# ### Single 629 feature experiment on training data

# In[14]:


x_train_one_feature = x_train[629]
x_train_one_feature = x_train_one_feature.values.reshape(-1, 1)
logistic_regression.fit(x_train_one_feature, y_train.ravel())

y_res = logistic_regression.predict(x_train_one_feature)
print(accuracy_score(y_train, y_res))
print(classification_report(y_train, y_res))


# ### Plots confussion matrix for 629 results

# In[15]:


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["green", "red"])
    plt.yticks([0, 1], ["green", "red"])
    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')

plot_parameters(5, 5)
cm = confusion_matrix(y_train, y_res)
np.set_printoptions(precision=1)
plot_confusion_matrix(cm)


# ### Testing with four different features on test data - used in report

# In[16]:


def experiment_different_features(x_test, y_test, x_train, y_train, feature_indexes):
    accuracies = []
    
    for i in feature_indexes:
        # train on training set 
        x_one = x_train[i].values.reshape(-1, 1)
        logistic_regression.fit(x_one, y_train.ravel())
        
        # predict on testing set         
        x_one_test = x_test[i].values.reshape(-1, 1)
        y_res = logistic_regression.predict(x_one_test)
        accuracies.append({i: accuracy_score(y_test, y_res)})

    return accuracies


# In[17]:


accuracies = experiment_different_features(x_test, y_test, x_train, y_train, [300, 400, 600, 700])
accuracies


# ### Final results

# In[18]:


# train on training set
x_400 = x_train[400].values.reshape(-1, 1)
final_model = logistic_regression.fit(x_400, y_train.ravel())

to_clasify = pd.read_csv('../binary/XToClassify.csv', header=None)
to_clasify = to_clasify[400]
x_to_clasify_one = to_clasify.values.reshape(-1, 1)
y_final_res = final_model.predict(x_to_clasify_one)

print(y_final_res)
def write_to_file(df, path):
    df.to_csv(path, index=False, header=False, sep=",")

write_to_file(pd.DataFrame({'res': y_final_res}), "../binaryTask/PredictedClasses.csv")

