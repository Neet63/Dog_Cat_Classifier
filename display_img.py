#Display img
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

def Display(path):
    # display dog image
    img = mpimg.imread(path)
    imgplt = plt.imshow(img)
    plt.show()