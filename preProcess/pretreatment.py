import os
import csv
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
from numpy import expand_dims

def img_to_pixel(fl, size, gray):
    #  --- transform image to array and normalize it ---  #
    # @fl: the image file
    # @size: the size of image array
    # @gray: use gray or not

    image = Image.open(fl)
    if gray:
        image.convert('L')
    image = image.resize((size,size))
    data = np.array(image)
    data = data.reshape(size*size)
    data = data.astype('uint8')
    return data
    
def make_data(image_dir, save_dir, size, gray):
    #  --- make image data to array and save csv ---  #
    # @image_dir: the image dir 
    # @save_dir: the array data save dir
    # @size: the image size wanted to transform
    # @gray: use gray or not

    label_dict={
        'science': 0, # 科院
        'manage': 1, # 管院
        'edu': 2, # 教院
        'human': 3, # 人院
        'admin': 4, # 行政大樓
        'library': 5, # 圖書館
        'activity': 6, # 學活
        'restaurant': 7, # 學餐
        'restrauant': 7, # 學餐
    }

    with open(save_dir+'data_'+str(size)+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        pixel_list = []
        for i in range(size*size):
            pixel_list.append("pixel"+str(i))
        writer.writerow(['label']+pixel_list)
        for path in image_dir:
            files = os.listdir(path)
            for fl in files:
                pixels = img_to_pixel(path+fl, size, gray)
                label = label_dict[fl.split('_')[0]]
                writer.writerow([str(label)]+pixels.tolist())
    csvfile.close()

if __name__ == '__main__':
    make_data(image_dir=['../data/label_building/', '../data/augmentation/'], save_dir='../data/array/', size = 128, gray=True)