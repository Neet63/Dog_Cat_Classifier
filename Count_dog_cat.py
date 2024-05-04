import os
def countDog_Cat(path):
    #Counting number of dog and cat images
    file_names = os.listdir(path)

    dog_count = 0
    cat_count = 0

    for img_file in file_names:

        name = img_file[0:3]

        if name == 'dog':
            dog_count += 1

        else:
            cat_count += 1

    print('Number of dog images =', dog_count)
    print('Number of cat images =', cat_count)

    return dog_count,cat_count