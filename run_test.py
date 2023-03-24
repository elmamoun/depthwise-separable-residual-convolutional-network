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
                        help = 'directory to parse input images for testing')
    parser.add_argument('--model_path', type = str, default= 'models/best_model.pt')
    parser.add_argument('--batch_size', type = int , default = 1,
                         help= 'number of batch sizes for the corresponding Dataloader' )
    args = parser.parse_args()

    # Load the data
    try : 
        dataset = datasets.ImageFolder(root=args.root_dir, transform=prepare_data)
    except : 
        raise ValueError('Invalid root directory')
    
    ## Load the testing data 

    test_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2)

    
    ## Load the model 
    best_model = MyModel()
    best_model.load_state_dict(torch.load(args.model_path))
    best_model.to(device)
    
    eval(best_model, test_loader)

