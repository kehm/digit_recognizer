import csv
import numpy as np
import matplotlib.pyplot as pt
from sklearn.tree import DecisionTreeClassifier
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


def run_decision_tree_classifier(x_train, y_train, x_test, y_test):
    """Trains a decision tree classifier"""
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    #pt.imshow(255-images_test[0], cmap='gray')
    #pt.show()
    print(clf.predict([x_test[0]]))


def main():
    """Main function"""
    data_labels = np.asarray(load_labels(), dtype='float64')
    data_images = np.asarray(load_images(), dtype='float64')
    #data_images = data_images.reshape(data_images.shape[0], 28, 28)
    images_train, images_test, labels_train, labels_test = train_test_split(data_images, data_labels, test_size=0.25,
                                                                            random_state=42)
    run_decision_tree_classifier(images_train, labels_train, images_test, labels_test)


if __name__ == '__main__':
    main()
