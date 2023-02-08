import os
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split #משמש לחלוקה לטריין וטסט


def load_data():
    # build and an array of images and their classification
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    x_values = []
    y_values = []


    IMAGE_SIZE = (224, 224, 3)
    count = 0
    for path in  ["C:\\Users\\shell\\Downloads\\mushroomDataSet\\Mushrooms",
                  "C:\\Users\\shell\\Downloads\\mushroomDataSet2\\Mushrooms"]:

        for mushroom_type in os.listdir(path):
            if count >= 14:
                break

            mushroom_dir = os.path.join(path, mushroom_type)
            for file in os.listdir(mushroom_dir):
                count += 1
                if count > 14:
                    break
                print(f"Reading file # {count}...")

                #לקרוא את הקובץ לתוך משתנה בעזרת imrid
                full_path_filename = os.path.join(mushroom_dir, file)
                image = imread(full_path_filename)
                image = resize(image, IMAGE_SIZE)

                x_values.append(image)
                y_values.append(mushroom_type)

        x_train, y_train, x_test, y_test = train_test_split(x_values, y_values, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    print(len(x_train))
    print(len(x_test))
 #   test_labels.append(labels[mushrooms])
 #   print(test_labels)

    # train


    # test
