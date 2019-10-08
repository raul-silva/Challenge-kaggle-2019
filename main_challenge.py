# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:20:31 2019

@author: Raul Alfredo de Sousa Silva
"""

#%% Printing versions
print("Library versions used in this work:")
print("csv version: 1.0")
print("numpy version: 1.15.4")
print("Tensorflow version:1.13.1")
print("matplotlib version: 3.0.2")
print("sklearn version: 0.20.1")
#%%
import csv
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  cross_val_score, GridSearchCV, KFold

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


#%%
# Code from scikit-learn
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="red")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
class_names = ["healthy","cancer"]
#%% Writing predictions

def generate_file(y,filename):
    with open('features_test.csv', 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        A = []
        for row in csv_reader:
            A.append(row["ImageID"])

    labels = [['ImageId','Malignant']]
    for i in range(0,300):
        labels.append([A[i],int(y[i])])
        
    with open(filename, 'w',newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerows(labels)

#%% Reading features

##############################################################################
'''
Observation: Features were obtained based on the two other .py files delivered
with this one. It's a very long procedure even if I did all I could to 
accelerate it in terms of vectorization of the code. It took me around 17 hours
to finish so I do not recommend you to repeat, but if you really would like to
evaluate it you can uncomment the next lines below to turn it.
If not, just continue, the file with the featured should be attached with this 
code too.
'''
#import main_features as mf
## Change this to access training set images
#filename1 = 'C:\\Users\\raul-\\Documents\\DataSet-Challenge-IMA205-2019\\data\\im\\'
## Change this to access test set images (they can eventually be in different folders)
#filename2 = 'C:\\Users\\raul-\\Documents\\DataSet-Challenge-IMA205-2019\\data\\im\\'
## Change this to access training set names
#address1 = 'C:\\Users\\raul-\\Documents\\DataSet-Challenge-IMA205-2019\\data\\train.csv'
## Change this to access test set names
#address2 = 'C:\\Users\\raul-\\Documents\\DataSet-Challenge-IMA205-2019\\data\\test.csv'
## Change this to rename the training set features file
#featurefile1 = 'features_train_t.csv'
## Change this to rename the test set features file
#featurefile2 = 'features_test_t.csv'
#
## A reduction is applied to images in which one of the dimension is greater
## then 3000 pixels by default (to reduce the computational time). You are able
## to change it.
## Threshold for reduction
#lim = 3000
## New dimensions in case of reduction
#a=0 # size of rows
#b=0 # size of columns
## If they were let at 0 the default reduction will be applied, which means
## each dimension divided by 2.
#mf.features(filename1,filename2,address1,address2,featurefile1,featurefile2,
#            lim=3000,a=0,b=0)

##############################################################################

with open('features_train.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    row_count = 0
    A = []
    for row in csv_reader:
        for element in row:
            if line_count ==0:
                row_count+=1
            if row[element].find('IM') < 0:
                A.append(float(row[element]))
        line_count+=1
        
with open('features_test.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    row_count = 0
    for row in csv_reader:
        for element in row:
            if line_count ==0:
                row_count+=1
            if row[element].find('IM') < 0:
                A.append(float(row[element]))
        line_count+=1
        

X = np.reshape(A,(1000,(row_count-1)))
#%% Reading classes

with open('train.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    row_count = 0
    B = []
    for row in csv_reader:
        for element in row:
            if line_count ==0:
                row_count+=1
            if row[element].find('IM') < 0:
                B.append(int(row[element]))
        line_count+=1
y = np.reshape(B,(700))
#%%
##############################################################################
#                                Preprocessing                               #
##############################################################################

# Scale data (each feature will have average equal to 0 and unit variance)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Use number of components take explain 95% of variability
pca = decomposition.PCA(n_components=0.95)
pca.fit(X)
X_pca = pca.transform(X)  


X_train = X_pca[:700,:]
X_test = X_pca[700:,:]

#%%
##############################################################################
#                              First test: KNN                               #
##############################################################################

# Direct training
 
print("Fitting K-nearest neighbour to the training set")
p_grid_KNN = {'n_neighbors': [1,3,5,7,9,11,13,15]}
KNN = KNeighborsClassifier()
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid_KNN = GridSearchCV(estimator=KNN, param_grid=p_grid_KNN,
                        scoring='balanced_accuracy', cv=5)
grid_KNN.fit(X_train, y)
print("Best training Score: {}".format(grid_KNN.best_score_))
print("Best training params: {}".format(grid_KNN.best_params_))

y_pred = grid_KNN.predict(X_test)
name = 'test_KNN.csv' 
generate_file(y_pred,name)

#%%
##############################################################################
#                       Applying boosting to knn                             #
##############################################################################
# Creating multiple predictors (21)
p_grid_KNN = {'n_neighbors': [1,3,5,7,9,11,13,15]}
KNN = KNeighborsClassifier()

T = 21 
m = len(X_train)
ys = y - (y==0)
p = 1/m*np.ones(m)
index = np.zeros(400)
alpha = np.zeros(T)
grid_KNN = []
S = X_train
for t in range(T):
    Z = sum(p)
    p /= Z
    pcum = np.cumsum(p)
    for i in range(400):
        index[i] = np.where(pcum >= np.random.rand())[0][0]
    index = index.astype(np.int)
    St = X_train[index,:]
    yt = ys[index]
    grid_KNN.append(GridSearchCV(estimator=KNN, param_grid=p_grid_KNN,
                                 scoring='balanced_accuracy', cv=5))
    grid_KNN[-1].fit(St, yt)
    et = sum((grid_KNN[-1].predict(St)!=yt))/len(St)
    alpha[t] = 0.5*np.log((1-et)/(et+1e-10))
    correct1 = (grid_KNN[-1].predict(S) != ys).astype(np.int)
    correct2 = (grid_KNN[-1].predict(S) == ys).astype(np.int)
    correct = correct1 - correct2
    p = p/Z*np.e**(correct*alpha[t])
    
y_pred = np.zeros(300)
for t in range(T):
    y_pred += alpha[t]*grid_KNN[t].predict(X_test)
y_pred_fin = (y_pred > 0)
y_pred_fin = y_pred_fin.astype(np.int)

name = 'test_KNNboost.csv' 
generate_file(y_pred_fin,name)


#%%

##############################################################################
#                             Second test: SVM                               #
##############################################################################

# Fitting Linear SVM on original data
print("Fitting Linear SVM to the training set")

p_grid_lsvm = {'C': [1e-3,1e-2,1e-1,1,2,3,4,5,6,7,8,9,1e1]}
Lsvm = LinearSVC(class_weight='balanced')
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
# Nested CV with parameter optimization
clf = GridSearchCV(estimator=Lsvm, param_grid=p_grid_lsvm, cv=inner_cv)
nested_score = cross_val_score(clf, X_train, y, cv=outer_cv,
                               scoring='balanced_accuracy')

print("Average and std Nested Cv score : {0} +- {1}".format(nested_score.mean(),
      nested_score.std() ))

# Looking for the best hyperparameters
grid_lsvm = GridSearchCV(estimator=Lsvm, param_grid=p_grid_lsvm,
                         scoring='balanced_accuracy', cv=5)
grid_lsvm.fit(X_train, y)
print("Best Score: {}".format(grid_lsvm.best_score_))
print("Best params: {}".format(grid_lsvm.best_params_))

y_pred = grid_lsvm.predict(X_test)
name = 'test_SVM.csv' 
generate_file(y_pred,name)

#%%
##############################################################################
#                       Applying non_linearity to SVM                        #
##############################################################################
# Fitting Non-linear SVM
print("Fitting Non-linear SVM to the training set")
p_grid_nlsvm = {'C': [1e-3,1e-2,1e-1,1,2,3,4,5,6,7,8,9,1e1],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
NLsvm = SVC(kernel='rbf', class_weight='balanced')
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Looking for the best hyperparameters
grid_nlsvm = GridSearchCV(estimator=NLsvm, param_grid=p_grid_nlsvm,
                          scoring="balanced_accuracy", cv=5)
grid_nlsvm.fit(X_train, y)
print("Best Score: {}".format(grid_nlsvm.best_score_))
print("Best params: {}".format(grid_nlsvm.best_params_))


y_pred = grid_nlsvm.predict(X_test)
name = 'test_SVM_nl.csv' 
generate_file(y_pred,name)
#%%
##############################################################################
#                        Third test: Decision Trees                          #
##############################################################################
# Fitting Decision Trees
Tree = DecisionTreeClassifier()
Tree.fit(X_train,y)
# Score in the training set
print('Score in the training set is {0}'.format(Tree.score(X_train,y)) )

y_pred = Tree.predict(X_test)
name = 'test_tree.csv' 
generate_file(y_pred,name)

#%%
##############################################################################
#                               Fourth test: MLP                             #
##############################################################################
y_train = tf.keras.utils.to_categorical(y)

def init_weights_and_biases(shape, stddev=0.1, seed_in=None):
    """
    This function should return Tensorflow Variables containing the initialised
    weights and biases of the network,
    using a normal distribution for the initialisation, with stddev of the 
    normal as an input argument
    
    Parameters
    ----------
    shape : tuple, (n_input_features,n_hidden_1, n_hidden_2,n_classes)
        sizes necessary for defining the weights and biases
    
    Returns
    -------
    w1, b1, w2, b2, w3, b3: initialised weights and biases, with correct shapes
    """
    
    # BEGIN STUDENT CODE
    w1 = tf.Variable(tf.random_normal([shape[0],shape[1]], stddev=stddev,
                                      seed=seed_in))
    w2 = tf.Variable(tf.random_normal([shape[1],shape[2]], stddev=stddev,
                                      seed=seed_in))
    w3 = tf.Variable(tf.random_normal([shape[2],shape[3]], stddev=stddev,
                                      seed=seed_in))
    b1 = tf.Variable(tf.zeros([shape[1]]))
    b2 = tf.Variable(tf.zeros([shape[2]]))
    b3 = tf.Variable(tf.zeros([shape[3]]))
    # END STUDENT CODE
    return w1, b1, w2, b2, w3, b3

def forward_prop_multi_layer(X, w1, b1, w2, b2, w3, b3):
    """
    This function should define the network architecture, explained above
    
    Parameters
    ----------
    X : input to network
    w1, w2, w3 : Tensorflow Variables
        network weights
    b1, b2, b3 : Tensorflow Variables
        network biases 
    
    Returns
    -------
    Y_pred :
    the output layer of the network, the classification prediction
    """
    
    # BEGIN STUDENT CODE
    A1 = tf.nn.sigmoid(tf.matmul(X, w1)+b1)
    A2 = tf.nn.softmax(tf.matmul(A1,w2)+b2)
    Y_pred = tf.nn.softmax(tf.matmul(A2,w3)+b3)
    # END STUDENT CODE
    return Y_pred

def accuracy(Y_pred, Y_true):
    """
    This function calculates the network's accuracy, ie the percentage of 
    correct classifications.
    Here, we consider the class with the highest score in the network's output
    
    Parameters
    ----------
    Y_pred : Tensorfow variable
        predicted classification of network
    Y_true : Tensorfow variable
        true classes of input
    
    """
    
    acc = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_true, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    
    return acc

RANDOM_SEED = 52#42#
tf.set_random_seed(RANDOM_SEED)

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = X_train.shape[1]
n_classes = y_train.shape[1] # MNIST total classes (0-9 digits)

# tf Graph input
X_input = tf.placeholder("float", [None, n_input])
Y_true = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
stddev = 0.1

w1,b1,w2,b2,w3,b3 = init_weights_and_biases([n_input, n_hidden_1, n_hidden_2,
                                n_classes], stddev=0.1, seed_in=RANDOM_SEED)

# Construct model
Y_pred = forward_prop_multi_layer(X_input,w1,b1,w2,b2,w3,b3)

# Define loss and optimizer
cross_entropy = -tf.reduce_sum(Y_true * tf.log(Y_pred),axis=1)

loss = tf.reduce_mean(cross_entropy)
acc = accuracy(Y_pred, Y_true)

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_variables = optimizer.minimize(loss)

# Parameters
n_epochs = 500
train_accuracy = []
test_accuracy = []
batch_size = 700
display_step = 1
n_batches = int(np.ceil(X_train.shape[0]/batch_size))
print(n_batches)


with tf.Session() as sess:
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(n_epochs):
        # Loop over all batches
        for batch_idx in range(n_batches):
            #get the next batch in the MNIST dataset and carry out training
            poss = batch_size*batch_idx
            posf = batch_size*(batch_idx+1)
            
            #BEGIN STUDENT CODE
            sess.run([training_variables,loss], feed_dict={X_input:
                X_train[poss:posf,:], Y_true: y_train[poss:posf]})
            #END STUDENT CODE
            
        # calculate accuracy for this epoch
        train_accuracy.append(sess.run(acc, feed_dict={X_input: X_train,
                                                       Y_true:y_train}))
        u = sess.run([Y_pred], feed_dict={X_input: X_test})
        print(".", end='')
    
print("Training finished")


ypred = np.array(u)
ypred = ypred.reshape(300,2)
ypred = ypred>0.5
ypred = ypred[:,1]
print(sum(ypred.astype(np.int)))
name = 'test_MLP.csv' 
generate_file(ypred,name)

