import os
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split  # משמש לחלוקה לטריין וטסט
from matplotlib import pyplot
import numpy as np  # linear algebra


def show_example(image, label):
    pyplot.imshow(image)
    pyplot.title(f"label is {label} type is {mushroom_type_number2mushroom_type[label]}")
    pyplot.show()


def load_data():
    # build and an array of images and their classification

    x_values = []
    y_values = []

    IMAGE_SIZE = (224, 224, 3)
    for path in ["C:\\Users\\shell\\Downloads\\mushroomDataSet\\Mushrooms",
                 "C:\\Users\\shell\\Downloads\\mushroomDataSet2\\Mushrooms"]:

        mushroom_type_number = 0
        for mushroom_type in os.listdir(path):
            mushroom_type_number2mushroom_type.append(mushroom_type)
            if mushroom_type_number >= 5:
                break

            count = 0
            mushroom_dir = os.path.join(path, mushroom_type)
            for file in os.listdir(mushroom_dir):
                count += 1
                if count > 90:
                    break
                print(f"\rReading file # {count} {mushroom_type} {file}...", end="")

                # לקרוא את הקובץ לתוך משתנה בעזרת im-read
                full_path_filename = os.path.join(mushroom_dir, file)
                image = imread(full_path_filename)
                image = resize(image, IMAGE_SIZE)

                x_values.append(image)
                y_values.append(mushroom_type_number)
            mushroom_type_number += 1

        print()
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    x_train, y_train, x_test, y_test = train_test_split(x_values, y_values, test_size=0.33, random_state=42)

    print(f"Number of images in x train is:{len(x_train)}")
    print(f"Number of images in x test is:{len(x_test)}")
    print(f"Number of classes= {mushroom_type_number}")
    print(f"x_train average value for pixel= {np.average(x_train)}")
    print(f"x_train std value for pixel= {np.std(x_train)}")
    print(f"x_train Max value for pixel= {np.max(x_train)}")
    print(f"x_train min value for pixel= {np.min(x_train)}")
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    mushroom_type_number2mushroom_type = []
    x_train, y_train, x_test, y_test = load_data()

    # show_example(x_test[10], y_test[10])

    import tensorflow as tf
    keras = tf.keras
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

    tf.random.set_seed(1)
    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(5, 5), input_shape=x_train[0].shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(5, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))

#  test_labels.append(labels[mushrooms])
#  print(test_labels)

#  train
#  i = 5
#  print("i is {i}")


#  test
