import networkx as nx
import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import csv
import random

file_train = open ('C:\\Users\\vincent916735\Desktop\\Machine Learning\\train.txt')
file_test = open ('C:\\Users\\vincent916735\Desktop\\Machine Learning\\test-public.txt')
bigdata = open ('C:\\Users\\vincent916735\Desktop\\40w.txt')
mytrain = bigdata.readlines()
train = file_train.readlines()
test = file_test.readlines()
random.shuffle(mytrain)
del test[0]
g = nx.Graph()
connect = []
for i in range(20000):
    t = train[i].split()
    for j in range(len(t)):
        g.add_edge(t[0],t[j])

for i in range(19999):
    t1 = train[i].split()
    t2 = train[i+1].split()
    if g.has_edge(t1[0],t2[0]):
        connect.append([len(list(nx.common_neighbors(g,t1[0],t2[0]))),
            list(nx.adamic_adar_index(g, [(t1[0],t2[0])]))[0][2],
            list(nx.preferential_attachment(g, [(t1[0],t2[0])]))[0][2],
            list(nx.jaccard_coefficient(g,[(t1[0],t2[0])]))[0][2],
            list(nx.resource_allocation_index(g, [(t1[0],t2[0])]))[0][2],1])
    else:
        connect.append([len(list(nx.common_neighbors(g,t1[0],t2[0]))),
            list(nx.adamic_adar_index(g, [(t1[0],t2[0])]))[0][2],
            list(nx.preferential_attachment(g, [(t1[0],t2[0])]))[0][2],
            list(nx.jaccard_coefficient(g,[(t1[0],t2[0])]))[0][2],
            list(nx.resource_allocation_index(g, [(t1[0],t2[0])]))[0][2],0])


for i in range(300000):
    t0,t1,t2 = mytrain[i].split()
    connect.append([len(list(nx.common_neighbors(g,t0,t1))),
        list(nx.adamic_adar_index(g, [(t0,t1)]))[0][2],
        list(nx.preferential_attachment(g, [(t0,t1)]))[0][2],
        list(nx.jaccard_coefficient(g,[(t0,t1)]))[0][2],
        list(nx.resource_allocation_index(g, [(t0,t1)]))[0][2],t2])
    print(i)

for i in range(2000):
    t0,t1,t2 = test[i].split()
    connect.append([len(list(nx.common_neighbors(g,t1,t2))),
        list(nx.adamic_adar_index(g, [(t1,t2)]))[0][2],
        list(nx.preferential_attachment(g, [(t1,t2)]))[0][2],
        list(nx.jaccard_coefficient(g,[(t1,t2)]))[0][2],
        list(nx.resource_allocation_index(g, [(t1,t2)]))[0][2],0])
    print(i)


X = []
Y = []
X_train = []
Y_train = []
X_test = []
out = open('tr.csv','a', newline='')
csv_write = csv.writer(out,dialect='excel')

for i in range(len(connect)):
    line = connect[i]
    X.append([float(line[0]),float(line[1]),float(line[2]),float(line[3]),float(line[4])])
    Y.append(float(line[5]))
    csv_write.writerow(line)
out.close()

for i in range(len(X)-2000):
    X_train.append(X[i])
    Y_train.append(Y[i])
for i in range(len(X)-2000,len(X)):
    X_test.append(X[i])

sc = StandardScaler()
sc.fit(X_train)
X_train_stand = sc.transform(X_train)
sc.fit(X_test)
X_test_stand = sc.transform(X_test)


clf = MLPClassifier(activation = 'logistic',solver = 'sgd', hidden_layer_sizes=(50,50,50), random_state=1)
x,y = shuffle(X_train_stand,Y_train)
clf.fit(x,y)


tryit = open ('C:\\Users\\vincent916735\Desktop\\tryit.txt','w')
for i in range(len(X_test)): 
    tryit.write(str(clf.predict_proba(X_test_stand[i,:].reshape(1,-1))[0][1])+'\n')
tryit.close()