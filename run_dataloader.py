import argparse
from utils import * 

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in' , type = str, default = '/home/dataset_clean/', 
                        help = 'path to directory of input images')
    parser.add_argument('--path_out',type =str , default = '/home/dataset_clean_denoised/', 
                        help = 'path to directory of preprocessed images')
    parser.add_argument('--apply_preprocess' , type = bool, default= True,
                        help = "choose weither to apply preprocessing during creating training dataloader")

    args = parser.parse_args()


    ## Create a folder in local that contains images after denoising and enhancement
    classes = ['bcc','akiec','bkl','df','mel','nv','vasc']
    # classes = ['AK','BCC','BKL','DF','MEL','NV','SCC']
    alpha = 0.15
    beta = 1.5
    for c in classes : 
        dataloader( args.path_in + c + '/', args.path_out + c + '/' ,alpha,beta)