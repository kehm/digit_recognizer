import csv
import numpy as np


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


def main():
    """Main function"""
    data_labels = np.asarray(load_labels())
    data_images = np.asarray(load_images())
    data_images = data_images.reshape(data_images.shape[0], 28, 28)


if __name__ == '__main__':
    main()
