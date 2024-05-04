# #Extracting The dataset
# import Extract_zip_file
# dataset = Extract_zip_file.extractZipFile('E:\\ML\\Datasets\\train.zip')
# dataset2 = Extract_zip_file.extractZipFile('E:\\ML\\Datasets\\test.zip')

#Counting the number of files
import NumberOfFiles
NumberOfFiles.CountFiles('E:\\ML\\Projects\\ImageRecognition\\train')


#Importing Dependencies
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

#Displaying Dog and cat
from display_img import Display
# Display('E:\\ML\Projects\\ImageRecognition\\train\\dog.8298.jpg')

# Display('E:\\ML\\Projects\\ImageRecognition\\train\\cat.107.jpg')

#COunting number of dog and cat images
from Count_dog_cat import countDog_Cat
dog,cat = countDog_Cat('E:\\ML\\Projects\\ImageRecognition\\train')


#Resizing images
from Resize_img import resize_folder
# resize_folder('E:\\ML\\Projects\\ImageRecognition\\train\\', 
#               'E:\\ML\\Projects\\ImageRecognition\\image_resized\\')

countDog_Cat('E:\\ML\\Projects\\ImageRecognition\\image_resized\\')


#Creating labels for Cat and dog images
#Cat -> 0
#Dog -> 1
#Create label through for loop where images 1st 3 letter conating dog or cat
labels = []
file_names = os.listdir('E:\\ML\\Projects\\ImageRecognition\\image_resized')

for i in range(2084):
    
    file_name = file_names[i]
    label = file_name[0:3]

    if label=='cat':
        labels.append(0)
    else:
        labels.append(1)

print(len(labels))

#Counting dog and cat out of 2048 images
values,count = np.unique(labels, return_counts=True)
print(values,count)


#COnverting All resized images to numpyarray
import cv2
import glob
image_dir = 'E:\\ML\\Projects\\ImageRecognition\\image_resized\\'
image_extension = ['png','jpg']

files = []
[files.extend(glob.glob(image_dir + '*.' + e)) for e in image_extension]

dog_cat_images = np.asarray([cv2.imread(file) for file in files])
# print(dog_cat_images)
print(dog_cat_images.shape)


#Splliting features and labels
X = dog_cat_images
Y = np.asarray(labels)

#Train test split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,stratify=Y, random_state=5)

#Scaling the data
X_train_scaled = X_train/255
X_test_scaled = X_test/255


#Building Nueral Network
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# importing tensorflow and Keras
import tensorflow as tf 
tf.random.set_seed(3)
from tensorflow import keras
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,BatchNormalization,Dropout

# setting up the layers of Neural Network

model = keras.Sequential()

model.add(Conv2D(8,(5,5),strides=(1,1),activation='tanh',input_shape = (224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(16,(5,5),strides=(1,1),activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(32,(5,5),strides=(1,1),activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))


# compiling the Neural Network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



# training the Neural Network
history = model.fit(X_train_scaled, Y_train,batch_size = 50, validation_split=0.1, epochs=10)

#Visulizing Accuracy and Loss 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['training data', 'validation data'], loc = 'lower right')
plt.show()


#Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['training data', 'validation data'], loc = 'upper right')
plt.show()


#Prediction System
from prediction_system import predict
predict(model)

import pickle

# Save the trained model
# filename = 'dog_cat_classifier.sav'
# pickle.dump(model, open(filename, 'wb'))
model.save('dog_cat_classifier.h5')

