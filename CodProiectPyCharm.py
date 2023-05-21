"""
Documentatie folosita:
1. https://www.digitalocean.com/community/tutorials/how-to-build-a-machine-learning-classifier-in-python-with-scikit-learn
2. https://scikit-learn.org/stable/modules/tree.html
3. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

"""
import pandas
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# organizare date pe coloane
nume_coloane=['var','skew','curt','entr','class']

# citire setului de date
data=pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt', names=nume_coloane)

""""# extragere coloana cu class
data_clase = data.iloc[: ,4]
print(data_clase) 
print('\n')
"""

# impartim datele in seturi
train, test, train_labels, test_labels = train_test_split(data.iloc[:,:4], data.iloc[:,4], test_size=0.33)
print(train) # pentru verificare
print('\n')

# antrenarea modelului
tree= tree.DecisionTreeClassifier()
model=tree.fit(train, train_labels)
print(train_labels) # pentru verificare
print('\n') 

# realizarea predictilor
preds=tree.predict(test)
print(preds) # pentru verificare
print('\n')

# masurarea acuratetii
print('In urma masuratorii s-a obinut o acuratete de:' ,accuracy_score(test_labels,preds)*100, '%')
