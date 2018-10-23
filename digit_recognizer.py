import csv


def load_labels():
    """Returns array of labels"""
    labels = []
    with open("resources/handwritten_digits_labels.csv", mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            labels.append(row)
    return labels


def load_images():
    """Returns array of images"""
    images = []

    return images


def main():
    """Main function"""
    data_labels = load_labels()
    data_images = load_images()
    print(data_labels)


if __name__ == '__main__':
    main()
