import os
import json
import random
from PIL import Image
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def generate_img(img, N, rotateRange, wShiftRange, hShiftRange, zoomRange, brightRange):
    #  --- generate images ---  #
    # @img: the image will be dealed with
    # @N: number of generated 
    # @rotateRange: max degree of random rotate (int: 0~180)
    # @wShiftRange: width shift size (float: 0~1)
    # @hShiftRange: height shift size (float: 0~1)
    # @zoomRange: the range for random zoom (float or [LB, UB] for width and height)
    # @brightRange: the range for brightness(tuple or list for two float)

    # generate image
    data = img_to_array(img)
    samples = expand_dims(data, 0)
    datagen = ImageDataGenerator(rotation_range=rotateRange, width_shift_range=wShiftRange, height_shift_range=hShiftRange, zoom_range=zoomRange, brightness_range=brightRange)
    it = datagen.flow(samples, batch_size=1)
    newImages = []
    for i in range(N):
        batch = it.next()
        newImage = batch[0].astype('uint8')
        newImages.append(newImage)
    return newImages


def augment_img(files, augN, genN, re_size):
    #  --- augment images ---  #
    # @files: origin image file 
    # @augN: random augment n image
    # @genN: generate n images for 7 method
    
    # generate new images for every files
    flag = False
    aug_method = []
    for rotate_range in [0, 80]:
        for wShift_range in [0, 0.3]:
            for zoom_range in [0, 0.7]:
                for bright_range in [None, (0.6, 1.4)]:
                    if flag:
                        aug_method.append([rotate_range, wShift_range, zoom_range, bright_range])
                    flag = True
    
    gen_images = []
    for fl in files:
        print(fl, len(gen_images))
        img = Image.open(fl)#.convert("RGB")
        image = img.resize(re_size)
        for i in range(genN):
            seed = random.randrange(len(aug_method))
            gen_images.extend(generate_img(image, 1, rotateRange=aug_method[seed][0], wShiftRange=aug_method[seed][1], hShiftRange=0.0, zoomRange=aug_method[seed][2], brightRange=aug_method[seed][3]))
    random.shuffle(gen_images)
    return gen_images[:augN]


def search_files(path, keyword):
    #  --- search files which filename contains keyword at target directory ---  #
    # @path: the target directory we want to search
    # @keyword: the keyword searching at target directory

    filenames = os.listdir(path)
    match_files = [os.path.join(path,name) for name in filenames if keyword in name]
    return match_files


def add_images(image_dir, save_image_dir, requestN, origin):
    #  --- augment image to requestN and save to path  ---  #
    # @image_dir: the origin images used to augment image is in this directory
    # @save_image_dir: the directory where save augment image
    # @save_array_path: the directory where save array data
    # @requestN: the images in two directory reached this amount

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
    
    # augument images if images does not reach the amount requested
    for key in label_dict.keys():
        files = search_files(image_dir, key)
        haveN = len(files)
        if haveN == 0:
            continue
        augN = requestN-haveN
        genN = int(augN/haveN)+1
        re_size = (128,128)
        if origin:
            num = 0
            for fl in files:
                img = Image.open(fl).convert("RGB")
                image = img.resize(re_size)
                image.save(os.path.join(save_image_dir,'{0}_{1}.jpg'.format(key, num)))
                num += 1 
                if num > requestN:
                    break
        if augN > 0:
            aug_images = augment_img(files, augN, genN, re_size)
            # save images
            num = haveN
            for image in aug_images:
                plt.imsave(os.path.join(save_image_dir,'{0}_{1}.jpg'.format(key, num)), image)
                num += 1           
        

if __name__ == '__main__':
    add_images(image_dir='../data/label_building/', save_image_dir='../data/aug_new/', requestN=10000, origin=True)
