import numpy as np
from itertools import chain
import math
import pandas as pd 
import matplotlib.pyplot as plt
import time

st_time = time.time()
data = pd.read_csv('haberman.csv')

mylist = list(data['class'].unique())

df_norm = data[['Age', 'Year', 'pos_aux_node']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
target = data[['class']].replace([1,2],[0,1])

df = pd.concat([df_norm, target], axis=1)

train_test_per = 80/100.0
df['train'] = np.random.rand(len(df)) < train_test_per

train1 = df[df.train == 1]
train = train1.drop('train', axis=1).sample(frac=1)

In_train = train.values[:,:3]
y = np.array(train.values[:,3:4])


test = df[df.train == 0]
test = test.drop('train', axis=1).sample(frac=1)

In_test = test.values[:,:3]
y1 = np.array(test.values[:,3:4])

from sklearn.decomposition import FastICA
ica = FastICA(n_components = 2)
In_train = ica.fit_transform(In_train)
In_test = ica.transform(In_test)



num_inputs = len(In_train[0])
hidden_layer_neurons =  math.ceil((num_inputs + len(mylist))/2)
w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1
num_outputs = len(y[0])
w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1
#np.random.seed(1)
#learning_rate = 0.008744926050643462
learning_rate = np.random.uniform(0,0.01)
print(learning_rate)

er = [0]*1000
for epoch in range(1000):
    l1 = 1/(1 + np.exp(-(np.dot(In_train, w1)))) # sigmoid function
    l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))
    x = list(chain(*(abs(y - l2))))
    er[epoch] = pow((np.mean([math.sqrt(i) for i in x])),2)
    l2_delta = (y - l2)*(l2 * (1-l2))
    l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
    w2 += l1.T.dot(l2_delta) * learning_rate
    w1 += In_train.T.dot(l1_delta) * learning_rate

epoch1 = list(range(epoch+1))
ax = plt.subplot()
p1, =plt.plot(epoch1,er, '-', color='black', label= 'ANN', lw=2)
plt.xlabel("Number of Epochs") 
plt.ylabel("Error")
ax.legend()
plt.show()         

y1 = (y1==1)

l1 = 1/(1 + np.exp(-(np.dot(In_test, w1))))
l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))


yp = (l2>=0.5)
res = (yp == y1)
correct = np.sum(res)/len(res)

y1 = list(chain(*(y1)))
yp = list(chain(*(yp)))

y_actu = pd.Series(y1, name='Actual')
y_pred = pd.Series(yp, name='Predicted')
from sklearn.metrics import confusion_matrix
df_confusion = confusion_matrix(y_actu, y_pred)


prec = (df_confusion[1][1]/(df_confusion[1][1]+df_confusion[1][0]))
recall = (df_confusion[1][1]/(df_confusion[1][1]+df_confusion[0][0]))
f_measure = (2*prec*recall)/(prec+recall)

testres = test[['class']].replace([0,1], ['1','2'])
testres['Prediction'] = yp
testres['Prediction'] = testres['Prediction'].replace([0,1],['1','2'])

print(testres)
print('Correct:',sum(res),'/',len(res), ':', (correct*100),'%')
print('Error:',np.mean(er))
print('Precision:',prec)
print('Recall:',recall)
print('F-Measure:',f_measure)
print("Time: %s  seconds"%(time.time() - st_time))
time = time.time() - st_time