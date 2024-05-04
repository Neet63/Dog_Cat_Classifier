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
import tensorflow as tf
import tensorflow_hub as hub

# Load the MobileNetV2 model from TensorFlow Hub
mobilenet_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
pretrained_model = hub.KerasLayer(mobilenet_model, input_shape=(224,224,3), trainable=False)

# Create a Sequential model
model = tf.keras.Sequential()

# Add the pre-trained model as a layer to the Sequential model
model.add(pretrained_model)

# Add a Dense layer for classification (assuming 2 classes)
num_of_classes = 2
model.add(tf.keras.layers.Dense(num_of_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

score, acc = model.evaluate(X_test_scaled, Y_test)
print('Test Loss =', score)
print('Test Accuracy =', acc)


#Prediction System
from prediction_system import predict
predict(model)