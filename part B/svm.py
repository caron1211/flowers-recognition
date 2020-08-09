from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import PrepareData

train_features, train_labels, test_features, test_labels = PrepareData.getData()

# create the model - SVM
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(train_features, train_labels)
print('Score train: ', svm.score(train_features, train_labels))

print('Score test: ', svm.score(test_features, test_labels))
prediction = svm.predict(test_features)

print(classification_report(test_labels, prediction))
matrix = confusion_matrix(test_labels, prediction)
print(matrix)


