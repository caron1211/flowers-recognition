

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import PrepareData


train_features, train_labels, test_features, test_labels = PrepareData.getData()

# create the model - Random Forests
clf = RandomForestClassifier(n_estimators=50, random_state=9)
# clf = RandomForestClassifier()


# fit the training data to the model
clf.fit(train_features, train_labels)
print('Score train: ', clf.score(train_features, train_labels))
print('Score test: ', clf.score(test_features, test_labels))
prediction = clf.predict(test_features)

print(classification_report(test_labels, prediction))
matrix = confusion_matrix(test_labels, prediction)
print(matrix)
