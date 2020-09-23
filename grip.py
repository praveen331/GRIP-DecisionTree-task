
import sklearn.datasets as datasets
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.pyplot as plt

iris = datasets.load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
df.describe()
y=iris.target
print(y);

X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.2,random_state=1)


clf = DecisionTreeClassifier(criterion='entropy',max_depth=3)

clf = clf.fit(X_train,y_train)

ypred = clf.predict(X_test)

print("accuracy is :",metrics.accuracy_score(y_test,ypred))

import pickle
with open('model_grip','wb') as f:
    pickle.dump(clf,f)
    
with open('model_grip','rb') as f:
    np = pickle.load(f)
    
    print(np.predict([[2,1,1,0.2]]))

from sklearn import tree
tree.plot_tree(clf);
fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['iris-setosa', 'iris-versicolor', 'iris-virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')
