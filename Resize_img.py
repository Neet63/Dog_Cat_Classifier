from PIL import Image
import os

def resize_folder(originalpath,newpath):


    os.mkdir(newpath)

    original_dir = originalpath
    resized_dir = newpath

    for i in range(25000):
        if(i%12 == 0):
            file_name = os.listdir(original_dir)[i]
            img_path = original_dir+file_name

            img = Image.open(img_path)
            resized_img = img.resize((224,224))
            img = img.convert('RGB')

            new_img_path = resized_dir + file_name
            resized_img.save(new_img_path)
            # Display(new_img_path)