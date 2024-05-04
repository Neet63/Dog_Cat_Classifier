import cv2
import numpy as np
from display_img import Display

def predict(model):
    input_image_path = input('Path of the image to be predicted: ')

    input_image = cv2.imread(input_image_path)
    Display(input_image_path)
    # cv2.imshow(input_image)

    input_image_resize = cv2.resize(input_image, (224,224))

    input_image_scaled = input_image_resize/255

    image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])

    input_prediction = model.predict(image_reshaped)

    print(input_prediction)

    input_pred_label = np.argmax(input_prediction)

    print(input_pred_label)

    if input_pred_label == 0:
        print('The image represents a Cat')

    else:
        print('The image represents a Dog')