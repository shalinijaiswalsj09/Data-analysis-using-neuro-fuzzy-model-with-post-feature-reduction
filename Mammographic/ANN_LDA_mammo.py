import numpy as np
from itertools import chain
import math
import pandas as pd 
import matplotlib.pyplot as plt
import time

st_time = time.time()
data = pd.read_csv('Cleaned_data.csv')
df_norm = data[['BI-RADS', 'Age', 'Shape', 'Margin','Density']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

target = data[['Severity']]
df = pd.concat([df_norm, target], axis=1)

train_test_per = 80/100.0
df['train'] = np.random.rand(len(df)) < train_test_per

train1 = df[df.train == 1]
train = train1.drop('train', axis=1).sample(frac=1)

X = train.values[:,:5]
y = train.values[:,5:6]

test = df[df.train == 0]
test = test.drop('train', axis=1).sample(frac=1)


X1 = test.values[:,:5]
y1 = test.values[:,5:6]

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 5)
In_train = lda.fit_transform(X,y)
In_test = lda.transform(X1)
explained_variance = lda.explained_variance_ratio_


num_inputs = len(In_train[0])
hidden_layer_neurons = math.ceil((num_inputs+len(y[0]))/2)
w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1
num_outputs = len(y[0])
w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1
#np.random.seed(7)
learning_rate = 0.8840632175776466
#learning_rate = np.random.uniform(0.1,0.2)
print(learning_rate)

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

y1 = (y1==1)

l1 = 1/(1 + np.exp(-(np.dot(In_test, w1))))
l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

yp = (l2 >= 0.5) # prediction
res = (yp == y1)
correct = np.sum(res)/len(res)

y1 = list(chain(*(y1)))
yp = list(chain(*(yp)))

y_actu = pd.Series(y1, name='Actual')
y_pred = pd.Series(yp, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)

prec = (df_confusion[True][True]/(df_confusion[True][True]+df_confusion[True][False]))
recall = (df_confusion[True][True]/(df_confusion[True][True]+df_confusion[False][False]))
f_measure = (2*prec*recall)/(prec+recall)

testres = test[['Severity']].replace([0,1], ['0','1'])
testres['Prediction'] = yp
testres['Prediction'] = testres['Prediction'].replace([0,1], ['0','1'])

print(testres)
print('Correct:',sum(res),'/',len(res), ':', (correct*100),'%')
print('Error:',np.mean(er))
print('Precision:',prec)
print('Recall:',recall)
print('F-Measure:',f_measure)
print("Time: %s  seconds"%(time.time() - st_time))