import numpy as np
from itertools import chain
import math
import pandas as pd 
import matplotlib.pyplot as plt
import time

st_time = time.time()
data = pd.read_csv('ecoli.csv')
mylist = list(data['class'].unique())
df_norm = data[['mcg','gvh','lip','chg','aac','alm1','alm2']]
target = data[['class']].replace([0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7])

df = pd.concat([df_norm, target], axis=1)

train_test_per = 80/100.0
df['train'] = np.random.rand(len(df)) < train_test_per

train = df[df.train == 1]
train = train.drop('train', axis=1).sample(frac=1)

test = df[df.train == 0]
test = test.drop('train', axis=1)
 

X = train.values[:,:7]
targets = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]]
y = np.array([targets[int(x)] for x in train.values[:,7:8]])
y3 = list(map(int,(np.array(train.values[:,(len(train.columns)-1):len(train.columns)]))))


X1 = test.values[:,:7]
y1 = list(map(int,(np.array(test.values[:,7:8]))))
y2 = np.array([targets[int(x)] for x in test.values[:,7:8]])



from sklearn.decomposition import FastICA
ica = FastICA(n_components = 5)
In_train = ica.fit_transform(X)
In_test = ica.transform(X1)



num_inputs = len(In_train[0])
hidden_layer_neurons = math.ceil((num_inputs+len(y[0]))/2)

w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1
num_outputs = len(y[0])
w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1
#np.random.seed(6)
learning_rate = 0.08928
#learning_rate = np.random.uniform(0,0.5)
#print(learning_rate)

er = [0]*300
for epoch in range(300):
    l1 = 1/(1 + np.exp(-(np.dot(In_train, w1)))) # sigmoid function
    l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))
    x = list(chain(*(abs(y - l2))))
    er[epoch] = pow((np.mean([math.sqrt(i) for i in x])),2)
    l2_delta = (y - l2)*(l2 * (1-l2))
    l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
    w2 += l1.T.dot(l2_delta) * learning_rate
    w1 += In_train.T.dot(l1_delta) * learning_rate

epoch1 = list(range(epoch+1))
plt.plot(epoch1,er)
plt.xlabel("Number of Epochs") 
plt.ylabel("Error")
plt.title("Variation of Error with No. of Epochs (ANN with PCA)")
plt.show()         


l1 = 1/(1 + np.exp(-(np.dot(In_test, w1))))
l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

yp = np.argmax(l2, axis=1) # prediction
res = yp == np.argmax(y2, axis=1)
correct = np.sum(res)/len(res)

y_actu = pd.Series(y1, name='Actual')
y_pred = pd.Series(yp, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)

prec0 = (df_confusion[0][0]/(df_confusion[0][0]+df_confusion[0][1]+df_confusion[0][2]))
prec1 = (df_confusion[1][1]/(df_confusion[1][0]+df_confusion[1][1]+df_confusion[1][2]))
prec2 = (df_confusion[2][2]/(df_confusion[2][0]+df_confusion[2][1]+df_confusion[2][2]))
prec = ((prec0+prec1+prec2)/3)
recall0 = (df_confusion[0][0]/(df_confusion[0][0]+df_confusion[1][0]+df_confusion[2][0]))
recall1 = (df_confusion[1][1]/(df_confusion[0][1]+df_confusion[1][1]+df_confusion[2][1]))
recall2 = (df_confusion[2][2]/(df_confusion[0][2]+df_confusion[1][2]+df_confusion[2][2]))
recall = ((recall0+recall1+recall2)/3)
f_measure0 = (2*prec0*recall0)/(prec0+recall0)
f_measure1 = (2*prec1*recall1)/(prec1+recall1)
f_measure2 = (2*prec2*recall2)/(prec2+recall2)
f_measure = ((f_measure0+f_measure1+f_measure2)/3)

testres = test[['class']].replace([0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7])
testres['Prediction'] = yp
testres['Prediction'] = testres['Prediction'].replace([0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7])

print(testres)
print('Correct:',sum(res),'/',len(res), ':', (correct*100),'%')
print('Error:',np.mean(er))
print('Precision:',prec)
print('Recall:',recall)
print('F-Measure:',f_measure)
print("Time: %s  seconds"%(time.time() - st_time))