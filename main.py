import load_data

import matplotlib.pyplot as plt


def main():
    train_images, train_labels, train_bboxes, test_images, test_labels, test_bboxes = load_data.load_data()

    for train_image, train_label, train_bbox in zip(train_images, train_labels, train_bboxes):
        print("image_shape:", train_image.shape)
        print("label:", train_label)
        print("bbox:", train_bbox)
        """
            image_shape: (250, 376, 3)
            label: 198.1651376146789
            bbox: [120.549   30.0931 160.746  143.929 ]
        """
        break


if __name__ == "__main__":
    main()
