#import libraries for machine learning
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import scipy as sk
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

pd.options.mode.chained_assignment = None
#seperate my input file based on comma, and attribute the values in the following names
f2 = open('King-Rook-King.data.txt', 'r')
f1 = open('K-R-K.data.txt','w')
for line in f2:
        for ch in line[:12]:
            if (ch is 'a'):
                f1.write('9')
            elif (ch is 'b'):
                f1.write('10')
            elif (ch is 'c'):
                f1.write('11')
            elif (ch is 'd'):
                f1.write('12')
            elif (ch is 'e'):
                f1.write('13')
            elif (ch is 'f'):
                f1.write('14')
            elif (ch is 'g'):
                f1.write('15')
            elif (ch is 'h'):
                f1.write('16')
            else:
                f1.write(ch)
        f1.write(line[12:17])
f1.close()
f2.close()

#there was a problem with cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#it can not take string and i have to replace the columns with numbers


names = ['WhiteKingcolumn', 'WhiteKingline', 'WhiteRookcolumn', 'WhiteRookline', 'BlackKingcolumn','BlackKingline','Outcome']
data = pd.read_csv("new.txt",names=names )

#Checking my data
#data shape
print(data.shape)
print(data.head(20))

#This includes the count, mean, the min and max values as well as some percentiles
print(data.describe())


#class distribution
print(data.groupby('Outcome').size())
#Visualize my data


# box and whisker plots
data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# box and whisker plots
data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# scatter plot matrix
scatter_matrix(data)
plt.show()





# Split-out validation dataset
array = data.values
X = array[:,0:6]
Y = array[:,6]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms

#Implementation in order to measure algorithms' execution time
def compare_Algorithms():
    return model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

#this funtion's attributes is the speed and the acuraccy of every algorithm that the user have run
#attributes is an array with alla the speeds and acuraccies
def Hurwicz(names,times,results,a):
    decision = []
    i=0
    maxTime = 1#because I want to take the smallest time , but in the end the biggest decesion
    #so I make all the times 1-time , so i want to take the biggest of these elements
    for y in range (len(names)):
        x = a*(maxTime - times[y]) + (1-a)*results[y]
        decision.append(x)
    index, value = max(enumerate(decision), key=operator.itemgetter(1))
    print('The most suitable algorithm for you is:')
    print(names[index])

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results1 = []
names1 = []
times1 = []
results = []
names = []
time = []
print('Algorithm:     accuracy     time')
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = compare_Algorithms()
    if __name__ == "__main__":
        setup = "from __main__ import cv_results"
        time = timeit.timeit('cv_results', setup=setup)
    results.append(cv_results)
    names.append(name)
    msg = "%s:           %f      %f" % (name, cv_results.mean(),time)
    results1.append(cv_results.mean())
    names1.append(name)
    times1.append(time)
    print(msg)
print('Give me how much you prefer speed from acuraccy (with 1 you prefer only speed and with 0 only acuraccy)')
a = input()
Hurwicz(names1,times1,results1,a)




# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()




# Make predictions on validation dataset
#i take the algorithm with the biggest accuracy KNN
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
