from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

digits = load_digits()
# print(digits.data.shape)
# print(digits.data.size)
# print(digits.data.ndim)
# print(dir(digits))
# print(digits.data[0])
# plt.gray()
# for i in range(5):
#     plt.matshow(digits.images[i])
#     plt.show()

X = digits.data
Y = digits.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)

# print(Y_train.shape)
# print(Y_test.shape)

model = LogisticRegression()
model.fit(X_train,Y_train)

print('target value of the test', digits.target[1780])

result = model.predict([digits.data[1780]])

print('test result', result)

accuracy = model.score(X_test,Y_test)
print('model accuracy', accuracy)


Y_predicted = model.predict(X_test)
confusion = confusion_matrix(Y_test, Y_predicted)

# print(confusion)

ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test)
plt.show()