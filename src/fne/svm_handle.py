import numpy as np
from sklearn.svm import LinearSVC


class SVMHandle:
    def __init__(self):
        pass

    @staticmethod
    def train_svm_with_features(features_train, train_labels, features_test, test_labels):
        clf = LinearSVC()
        clf.fit(X=features_train, y=train_labels)
        predicted_labels = clf.predict(features_test)
        accuracy = (np.sum(test_labels == predicted_labels) / len(test_labels))
        return accuracy
