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
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.utils import np_utils


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
    clf = DecisionTreeClassifier()
    start_train = time.time()
    clf.fit(X_train, y_train.ravel())
    end_train = time.time()
    y_predict = clf.predict(X_test)
    end_test = time.time()
    score = accuracy_score(y_test, y_predict)
    print(clf.predict([X_test[0]]))
    print("Decision Tree Classifier Accuracy:", score, "Time spent training:", end_train - start_train,
          "Time spent testing:", end_test - end_train)


def run_random_forest_classifier(X_train, y_train, X_test, y_test):
    """Trains and tests a random forrest classifier"""
    clf = RandomForestClassifier()
    start_train = time.time()
    clf.fit(X_train, y_train.ravel())
    end_train = time.time()
    y_predict = clf.predict(X_test)
    end_test = time.time()
    score = accuracy_score(y_test, y_predict)
    print(clf.predict([X_test[0]]))
    print("Random Forest Classifier Accuracy:", score, "Time spent training:", end_train - start_train,
          "Time spent testing:", end_test - end_train)


def run_k_neighbors_classifier(X_train, y_train, X_test, y_test):
    """Trains and tests a K nearest neighbors classifier"""
    clf = KNeighborsClassifier()
    start_train = time.time()
    clf.fit(X_train, y_train.ravel())
    end_train = time.time()
    y_predict = clf.predict(X_test)
    end_test = time.time()
    score = accuracy_score(y_test, y_predict)
    print(clf.predict([X_test[0]]))
    print("K Nearest Neighbors Classifier Accuracy:", score, "Time spent training:", end_train - start_train,
          "Time spent testing:", end_test - end_train)


def run_support_vector_machine(X_train, y_train, X_test, y_test):
    """Trains and tests a support vector machine"""
    clf = SVC(kernel='linear', C=1E10)
    start_train = time.time()
    clf.fit(X_train, y_train.ravel())
    end_train = time.time()
    y_predict = clf.predict(X_test)
    end_test = time.time()
    score = accuracy_score(y_test, y_predict)
    print(clf.predict([X_test[0]]))
    print("Support Vector Machine Accuracy:", score, "Time spent training:", end_train - start_train,
          "Time spent testing:", end_test - end_train)


def run_neural_network(X_train, y_train, X_test, y_test):
    """Trains and tests a neural network"""
    y_train = np_utils.to_categorical(y_train, 10)  # converting labels array to categorically instead of binary
    y_test = np_utils.to_categorical(y_test, 10)
    model = Sequential()
    model.add(Dense(512, input_shape=(784,), activation='relu'))
    model.add(Dropout(0.2))  # adding dropout to avoid overfitting
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    start_train = time.time()
    model.fit(X_train, y_train, epochs=8, batch_size=128, verbose=0, validation_data=(X_test, y_test))
    end_train = time.time()
    score = model.evaluate(X_test, y_test, verbose=0)
    end_test = time.time()
    print("Neural Network Accuracy:", score[1], "Time spent training:", end_train - start_train,
          "Time spent testing:", end_test - end_train)


def main():
    """Main function"""
    data_labels = np.asarray(load_labels(), dtype='float64')
    data_images = np.asarray(load_images(), dtype='float64')
    images_train, images_test, labels_train, labels_test = train_test_split(data_images, data_labels, test_size=0.25,
                                                                            random_state=42)
    show_image(images_test[0])
    images_train /= 255  # normalizing values
    images_test /= 255  # normalizing values
    run_decision_tree_classifier(images_train, labels_train, images_test, labels_test)
    run_random_forest_classifier(images_train, labels_train, images_test, labels_test)
    #run_k_neighbors_classifier(images_train, labels_train, images_test, labels_test)
    #run_support_vector_machine(images_train, labels_train, images_test, labels_test)
    run_neural_network(images_train, labels_train, images_test, labels_test)


if __name__ == '__main__':
    main()
