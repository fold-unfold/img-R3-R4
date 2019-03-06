import numpy as np
import imageio
from PIL import Image
import IPython.display
import pickle
import os

DATA_GENERATED_PATH = 'data_generated'
DATA_PATH = 'camera_relocalization_sample_dataset\camera_relocalization_sample_dataset'
data_file_txt = '%s\%s' % (DATA_PATH, 'info.csv')
data_path_img = '%s\%s' % (DATA_PATH, 'images')

data_num = np.genfromtxt(data_file_txt, dtype=(int, int), delimiter=',', skip_header=1, usecols=(0, 1))
data_timestamp = np.genfromtxt(data_file_txt, dtype=('|U'), delimiter=',', skip_header=1, usecols=(2))

data_pos = np.genfromtxt(data_file_txt, delimiter=',', skip_header=1, usecols=(3, 4, 5))
data_q = np.genfromtxt(data_file_txt, delimiter=',', skip_header=1, usecols=(6, 7, 8, 9))

data_imu_la = np.genfromtxt(data_file_txt, delimiter=',', skip_header=1, usecols=(10, 11, 12))
data_imu_av = np.genfromtxt(data_file_txt, delimiter=',', skip_header=1, usecols=(13, 14, 15))

data_file_names = np.genfromtxt(data_file_txt, dtype='|U', delimiter=',', skip_header=1, usecols=(16, 17, 18))

def readImg(file_name, img_path=data_path_img, ignore_ch_4=True):
    file_path = '%s\%s' % (img_path, file_name)
    image = imageio.imread(file_path)
    if ignore_ch_4:
        image = image[:,:,:3]
    return(image)

def showImg(image_np):
    img = Image.fromarray(image_np, 'RGB')
    IPython.display.display(img)
    
def getImgData(names=data_file_names[:, 0]):
    data_img = []
    for file_name in names:
        image = readImg(file_name)
        data_img.append(image)
    data_img = np.array(data_img)
    return(data_img)

def save_if_not_exist(data, file_name, data_dir = DATA_GENERATED_PATH):
    file_path = os.path.join(data_dir, file_name)

    if not os.path.exists(file_path):
        # Save the data to a cache-file.
        with open(file_path, mode='wb') as file:
            pickle.dump(data, file)

def load_if_exist(file_name, data_dir = DATA_GENERATED_PATH):
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        with open(file_path, mode='rb') as file:
            data = np.load(file)
    else:
        data = None
    
    return data