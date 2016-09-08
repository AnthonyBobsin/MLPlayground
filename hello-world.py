from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)

print(classifier.predict([[160, 0]]))

from sklearn.externals.six import StringIO
import pydotplus as pydot
dot_data = StringIO()
tree.export_graphviz(classifier, out_file=dot_data)
graphs = pydot.graph_from_dot_data(dot_data.getvalue())
graphs.write_pdf("hello-world.pdf")
