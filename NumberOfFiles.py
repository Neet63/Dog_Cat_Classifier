#Count the number of file in folder
import os
def CountFiles(Folder):
    # counting the number of files in train folder
    path, dirs, files = next(os.walk(Folder))
    file_count = len(files)
    print('Number of images: ', file_count)
    #Display name of files
    file_names = os.listdir(Folder)
    # print(file_names)