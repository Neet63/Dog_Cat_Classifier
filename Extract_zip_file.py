#Extracting the train compressed zip file
def extractZipFile(path):

    from zipfile import ZipFile
    dataset = path
    with ZipFile(dataset, 'r') as zip:
        zip.extractall()
        print(path, '  Dataset is Extracted')
    return dataset

