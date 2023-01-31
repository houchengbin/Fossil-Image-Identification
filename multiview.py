'''
multiview.py
create multi-view images of original images 
use for image augmentation
'''

import argparse
import cv2
import numpy as np
from skimage import morphology
import os


def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./data/Original_img',
                        help='input images path.')
    parser.add_argument('--output', type=str, default='./data',
                        help='output muti-view images path.')
    parser.add_argument('--gray', action="store_true", default=False,              
                        help='save gray_img flag: default False')
    parser.add_argument('--binary', action="store_true", default=False,               
                        help='binary flag: default False')
    parser.add_argument('--blocksize', type=int, default=41,                      
                        help='blocksize for binary')                          
    parser.add_argument('--C', type=float, default=5,              
                        help='C for binary') 
    parser.add_argument('--skeletonize', action="store_true", default=False,            
                        help='skeletonize model flag: default True')
    
    args = parser.parse_args()
    return args

def find_file(path):
    
    file_list=os.listdir(path)
    file_name_list = []
    
    for file_name in file_list:
            file_name_list.append(file_name)
            
    print(f'find {len(file_name_list)} files (out of {len(file_list)} all files) under {path}')
    return file_name_list


def find_img_file(path):
    
    file_list=os.listdir(path)
    img_name_list = []
    
    for file_name in file_list:

        if file_name[-4:]=='.png':
            img_name_list.append(file_name)
            
    print(f'find {len(img_name_list)} image files (out of {len(file_list)} all files) under {path}')
    return img_name_list


def multiview(args):
   
    img_files_list = find_file(args.input)   
    os.makedirs(args.output, exist_ok=True)

    for img_files_name in img_files_list:    
        img_classification_list = find_file(os.path.join(args.input, img_files_name))
        for img_classification in img_classification_list:
            img_name_list = find_img_file(os.path.join(args.input, img_files_name, img_classification))
            for img_name in img_name_list:
                
                img_origin_dir = os.path.join(args.input, img_files_name, img_classification, img_name)
                print(img_origin_dir)

                # ---- STEP 1 ---- read origin image and convert to gray img
                img_origin = cv2.imread(img_origin_dir)
                img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)     
            
                if args.gray:
                    
                    os.makedirs(os.path.join(args.output, 'gray', img_files_name, img_classification), exist_ok=True)
                    gray_dir = os.path.join(args.output, 'gray', img_files_name, img_classification)
                    cv2.imwrite(os.path.join(gray_dir, img_name), img_gray)
                    
                # ---- STEP 2 ---- convert to binary for skeletonize-process
                if args.binary:
                                        
                    # binary (black background)
                    blocksize = args.blocksize  
                    C = args.C                 
                    img_binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blocksize, C)
        
                # ---- STEP 3 ---- binary convert to skeleton 

                if args.skeletonize:

                    os.makedirs(os.path.join(args.output, 'skeleton', img_files_name, img_classification), exist_ok=True)
                    Skeleton_dir = os.path.join(args.output, 'skeleton', img_files_name, img_classification)
                        
                    img_skeletoniz = img_binary
                    img_skeletoniz[img_skeletoniz==255] = 1
                    skeleton0 = morphology.skeletonize(img_skeletoniz)
                    skeleton = skeleton0.astype(np.uint8)*255

                    # convert to white background and save skeleton image
                    skeleton_INV = skeleton
                    where_0 = np.where(skeleton_INV == 0)
                    where_255 = np.where(skeleton_INV == 255)
                    skeleton_INV[where_0] = 255
                    skeleton_INV[where_255] = 0
                    cv2.imwrite(os.path.join(Skeleton_dir, img_name), skeleton_INV)
                        

if __name__ == "__main__":
    args = get_args()
    print(args)
    multiview(args)

   
