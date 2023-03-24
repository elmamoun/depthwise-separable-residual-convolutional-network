import argparse
from models import * 
from utils import * 
import torchvision.datasets as datasets
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings 
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir' , type = str, default = '/home/dataset_clean_denoised/', 
                        help = 'directory to parse input images')
    parser.add_argument('--n_splits',type =int , default = 5, 
                        help = 'number of splits of RepeatedKfold')
    parser.add_argument('--plotting', action='store_true',  
                        help ='choose weither to plot epochs plots or not')
    parser.add_argument('--n_epochs' , type = int, default = 32,
                        help = 'number of epochs to train model')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help = 'Number of batch size for train and validation loaders')
    # parser.add_argument('--train_augmentation' , action = 'store_true', 
    #                     help = "Choose wheither to apply data augmentation in the training set" )

    args = parser.parse_args()
    
    



    # Load the data
    try : 
        dataset = datasets.ImageFolder(root=args.root_dir, transform=prepare_data)
        print('the size of the dataset is ', dataset.__len__())
    except : 
        raise ValueError('Invalid root directory')
    

    

    ## compute weights of each class 
    zeros = len([e for e in dataset.targets if e == 0])
    ones = len([e for e in dataset.targets if e == 1])
    twos = len([e for e in dataset.targets if e == 2])
    threes = len([e for e in dataset.targets if e == 3])
    fours = len([e for e in dataset.targets if e == 4])
    fives = len([e for e in dataset.targets if e == 5])
    sixes = len([e for e in dataset.targets if e == 6])

    weights = np.array([zeros, ones, twos, threes, fours, fives,sixes])
    weights = weights / sum(weights)
    weights = 1/weights
    weights = weights / sum(weights)
    
    model = MyModel().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight= torch.FloatTensor(weights).to(device))
    ## perform cross validation over the model 
    kfold_validation(model, dataset, n_sp = args.n_splits, bs = args.batch_size, 
                      plot = args.plotting, optim=  opt , n_ep= args.n_epochs , 
                      criterion= criterion)
    



