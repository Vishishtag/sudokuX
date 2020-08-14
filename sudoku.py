import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


#apt-get -qq install -y libsm6 libxext6 && pip install -q -U opencv-python


import matplotlib.pyplot as plt


from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
# load the data - it returns 2 tuples of digits & labels - one for
# the train set & the other for the test set
(train_digits, train_labels), (test_digits, test_labels) = load_data()

# display 14 random images from the training set
import numpy as np
np.random.seed(123)


image_height = train_digits.shape[1]
image_width = train_digits.shape[2]
num_channels = 1  # we have grayscale images
# NOTE: image_height == image_width == 28

# re-shape the images data
train_data = np.reshape(train_digits, (train_digits.shape[0], image_height, image_width, num_channels))
test_data = np.reshape(test_digits, (test_digits.shape[0],image_height, image_width, num_channels))

# re-scale the image data to values between (0.0,1.0]
train_data = train_data.astype('float32') / 255.
test_data = test_data.astype('float32') / 255.

# one-hot encode the labels - we have 10 output classes
# so 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0] & so on
from keras.utils import to_categorical
num_classes = 10
train_labels_cat = to_categorical(train_labels,num_classes)
test_labels_cat = to_categorical(test_labels,num_classes)
train_labels_cat.shape, test_labels_cat.shape






for _ in range(5):
    indexes = np.random.permutation(len(train_data))

train_data = train_data[indexes]
train_labels_cat = train_labels_cat[indexes]

# now set-aside 10% of the train_data/labels as the
# cross-validation sets
val_perc = 0.10
val_count = int(val_perc * len(train_data))

# first pick validation set from train_data/labels
val_data = train_data[:val_count,:]
val_labels_cat = train_labels_cat[:val_count,:]

# leave rest in training set
train_data2 = train_data[val_count:,:]
train_labels_cat2 = train_labels_cat[val_count:,:]

# NOTE: We will train on train_data2/train_labels_cat2 and
# cross-validate on val_data/val_labels_cat




from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model():
    model = Sequential()
    # add Convolutional layers
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                     input_shape=(image_height, image_width, num_channels)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    # output layer
    model.add(Dense(num_classes, activation='softmax'))
    # compile with adam optimizer & categorical_crossentropy loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()


from keras.models import load_model
model = load_model('./model.h5')



import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
#from google.colab import drive



from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
#from google.colab.patches import cv2_imshow
def gettingSquares(path):
    image = cv2.imread(path)
    height, width = image.shape[:2]
    matrix = np.zeros((9,9))
    for i in range(0, 9):
      for j in range(0, 9):
        cropped = image[(int)(height*i/9):(int)(height*(i+1)/9), (int)(width*j/9):(int)(width*(j+1)/9)]
        dim = (28,28)
        resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
        resized = resized[3:25, 3:25]
        resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        gray2 = gray.astype('float32')/255
        blank = gray2[9:19,9:19]
        isBlank = np.average(blank)
        blank = np.absolute(blank-isBlank)
        isBlank = np.sum(blank);

        if isBlank<=0.1:
          matrix[i][j]=0
        else:
          gray2 = gray2.reshape(1,28,28,1)
          prediction = model.predict(gray2)
          matrix[i][j] = (int)(np.argmax(prediction, axis=1))
    return matrix



def print_grid(arr):
    print (arr)
def find_empty_location(arr, l):
  for row in range(9):
    for col in range(9):
      if arr[row][col]== 0:
        l[0]= row
        l[1]= col
        return True
  return False
def used_in_row(arr, row, num):
	for i in range(9):
		if arr[row][i] == num:
			return True
	return False
def used_in_col(arr, col, num):
	for i in range(9):
		if arr[i][col] == num:
			return True
	return False
def used_in_box(arr, row, col, num):
	for i in range(3):
		for j in range(3):
			if arr[i + row][j + col] == num:
				return True
	return False
def check_location_is_safe(arr, row, col, num):
	return not used_in_row(arr, row, num) and not used_in_col(arr, col, num) and not used_in_box(arr, row - row % 3, col - col % 3, num)
def solve_sudoku(arr):
  l = [0,0]
  if find_empty_location(arr, l)==False:
    return True
  row = l[0]
  col = l[1]
  for num in range(1, 10):
    if check_location_is_safe(arr, row, col, num):
      arr[row][col]= num
      if solve_sudoku(arr):
        return True
      arr[row][col] = 0
  return False




from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
def solve(path):
    image = cv2.imread(path)
    dim = (280,280)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    sudoku = gettingSquares(path)
    #cv2.imshow(image)
    #print()
    #print(sudoku)
    #print()
    print(sudoku)
    solve_sudoku(sudoku)
    text = []
    for i in sudoku:
        temp = ""
        for j in i:
            temp = temp + str(int(j)) + " "
        text.append(temp)
    print (text)
    return text
#solve('S1.png')
