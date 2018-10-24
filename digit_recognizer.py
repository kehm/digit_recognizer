import csv
import time
import numpy as np
import matplotlib.pyplot as pt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_labels():
    """Returns array of labels"""
    labels = []
    file = "resources/handwritten_digits_labels.csv"
    with open(file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            labels.append(row)
    return labels


def load_images():
    """Returns array of images"""
    images = []
    file = "resources/handwritten_digits_images.csv"
    with open(file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            images.append(row)
    return images


def show_image(image_array):
    """Shows image"""
    image_array = image_array.reshape(28, 28)
    pt.imshow(255 - image_array, cmap='gray')
    pt.show()


def run_decision_tree_classifier(X_train, y_train, X_test, y_test):
    """Trains and tests a decision tree classifier"""
    start = time.time()
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train.ravel())
    y_predict = clf.predict(X_test)
    score = accuracy_score(y_test, y_predict)
    print(clf.predict([X_test[0]]))
    print("Decision Tree Classifier Accuracy:", score, "Running time:", time.time() - start)


def run_random_forest_classifier(X_train, y_train, X_test, y_test):
    """Trains and tests a random forrest classifier"""
    start = time.time()
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train.ravel())
    y_predict = clf.predict(X_test)
    score = accuracy_score(y_test, y_predict)
    print(clf.predict([X_test[0]]))
    print("Random Forest Classifier Accuracy", score, "Running time:", time.time() - start)


def run_k_neighbors_classifier(X_train, y_train, X_test, y_test):
    """Trains and tests a K nearest neighbors classifier"""
    start = time.time()
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train.ravel())
    y_predict = clf.predict(X_test)
    score = accuracy_score(y_test, y_predict)
    print(clf.predict([X_test[0]]))
    print("K Nearest Neighbors Classifier Accuracy", score, "Running time:", time.time() - start)


def run_support_vector_machine(X_train, y_train, X_test, y_test):
    """Trains and tests a support vector machine"""
    start = time.time()
    clf = SVC(kernel='linear', C=1E10)
    clf.fit(X_train, y_train.ravel())
    y_predict = clf.predict(X_test)
    score = accuracy_score(y_test, y_predict)
    print(clf.predict([X_test[0]]))
    print("Support Vector Machine Accuracy", score, "Running time:", time.time() - start)


def main():
    """Main function"""
    data_labels = np.asarray(load_labels(), dtype='float64')
    data_images = np.asarray(load_images(), dtype='float64')
    images_train, images_test, labels_train, labels_test = train_test_split(data_images, data_labels, test_size=0.25,
                                                                            random_state=42)
    #run_decision_tree_classifier(images_train, labels_train, images_test, labels_test)
    #run_random_forest_classifier(images_train, labels_train, images_test, labels_test)
    run_k_neighbors_classifier(images_train, labels_train, images_test, labels_test)
    #run_support_vector_machine(images_train, labels_train, images_test, labels_test)
    show_image(images_test[0])


if __name__ == '__main__':
    main()
