import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0, 50, 100]

# training data
training_target = np.delete(iris.target, test_idx)
training_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(training_data, training_target)

print(test_target)
print(classifier.predict(test_data))

from sklearn.externals.six import StringIO
import pydotplus as pydot
dot_data = StringIO()
tree.export_graphviz(classifier,
                                out_file=dot_data,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('pdf/iris.pdf')
