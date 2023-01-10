import os, random, shutil
import argparse
import os

def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./2400_fus',
                        help='input images path.')
    parser.add_argument('--output', type=str, default='./data/Original_img',
                        help='output split images path.')
    parser.add_argument('--train-rate', '--tr', type=float, default=0.734,
                        help='num = train/all.')
    parser.add_argument('--val-rate', '--vr', type=float, default=0.5,
                        help='num = val/(all-train).')
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
        
        if file_name[-4:]=='.png' or file_name[-4:]=='.jpg':
            img_name_list.append(file_name)
            
    print(f'find {len(img_name_list)} image files (out of {len(file_list)} all files) under {path}')
    return img_name_list

def moveFile(fileDir, tarDir, rate):
    
    pathDir = os.listdir(fileDir)  
    filenumber=len(pathDir)
    
    rate=float(rate)   
    picknumber=int(filenumber*rate)

    random.seed(2022)
    sample = random.sample(pathDir, picknumber)
    
    for name in sample:
        shutil.move(fileDir+name, tarDir+name)
    return

def datasplit(args):
    
    img_classification_list = find_file(args.input)   # classes
    os.makedirs(args.output, exist_ok=True)

    for img_classification in img_classification_list:     
        
        os.makedirs(os.path.join(args.output+'/train/', img_classification), exist_ok=True)
        fileDir = os.path.join(args.input, img_classification)+'/'
        tarDir = os.path.join(args.output+'/train/', img_classification)+'/'
        moveFile(fileDir, tarDir, rate = args.train_rate) 
        
    for img_classification in img_classification_list:
        
        os.makedirs(os.path.join(args.output+'/val/', img_classification), exist_ok=True)
        fileDir = os.path.join(args.input, img_classification)+'/'
        tarDir = os.path.join(args.output+'/val/', img_classification)+'/'
        moveFile(fileDir, tarDir, rate = args.val_rate)
        
    for img_classification in img_classification_list:
        
        os.makedirs(os.path.join(args.output+'/test/', img_classification), exist_ok=True)
        
        fileDir = os.path.join(args.input, img_classification)+'/'
        tarDir = os.path.join(args.output+'/test/', img_classification)+'/'
        moveFile(fileDir, tarDir, rate = 1)
        
        
if __name__ == "__main__":
    args = get_args()
    print(args)
    datasplit(args)
