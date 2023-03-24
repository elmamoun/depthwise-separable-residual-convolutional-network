import cv2
import torch 
import numpy as np 
import pywt
import os 
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from sklearn.model_selection import RepeatedKFold ,KFold
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader, random_split




def nlm_denoising(image) : 
    """ 
    Apply non local means denoising method to input image and returns denoised image
    """
    return cv2.fastNlMeansDenoisingColored(image, None, 2, 2, 15, 15)



def clahe_dwt(img,alpha, beta) : 

    """
    Apply CLAHE-DWT method to input image 
    """

    # Split the image into its RGB color channels
    r_channel, g_channel, b_channel = cv2.split(img)

    # Apply DWT and CLAHE to each color channel
    for i, channel in enumerate([r_channel, g_channel, b_channel]):
        
        # Retrieve the maximum and minimum intensity of original image 
        Iomin = channel.min()
        Iomax = channel.max()

        # Construsct H matrix used for weighted average 
        H = (channel - Iomin) / (Iomax - Iomin)
        H = torch.Tensor(H)
        H =  torch.pow(H, alpha).numpy()

        # Construct the final enchanced image
        IE = np.multiply(channel, H)
        Mones = np.ones(H.shape)


        coeffs = pywt.dwt2(channel, 'haar')
        cA, (cH, cV, cD) = coeffs
        
        # Apply CLAHE to the low frequency coefficients (cA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cA = clahe.apply(cA.astype(np.uint8))
        
        # Reconstruct the enhanced channel using the inverse DWT
        coeffs = (cA, (cH, cV, cD))
        channel_dwt = pywt.idwt2(coeffs, 'haar')
        
        # Take the weighted average 
        channel_weigh_updated = beta * np.multiply(channel_dwt ,( Mones - H))
        channel_weigh_updated = IE + channel_weigh_updated

        # Update the original image with the enhanced channel
        img[:,:,i] = channel_weigh_updated

    return img


def create_channels(img) :
    """
    Transform and create 6 output channels from corresponding image tensor
    """
    resize = transforms.Resize([384, 384],
                          interpolation=cv2.INTER_NEAREST)
    img = resize(img)

    # Convert PIL image to numpy array
    image = np.array(img)
    # Convert the image to different color spaces
    img_rgb = image
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_gray_inv = cv2.bitwise_not(img_gray)

    # Extract the saturation channel from the HSV image
    _, img_saturation, _ = cv2.split(img_hsv)

    # Extract the b* channel from the LAB image
    _, _, img_b = cv2.split(img_lab)

    # Construct the tensor with 6 channels
    tensor = np.zeros((image.shape[0], image.shape[1], 6), dtype=np.float32)
    tensor[:, :, 0:3] = img_rgb.astype(np.float32) 
    tensor[:, :, 3] = img_saturation.astype(np.float32) 
    tensor[:, :, 4] = img_b.astype(np.float32) 
    tensor[:, :, 5] = img_gray_inv.astype(np.float32)

    return tensor


def normalise(tensor) : 
    """
    Normalise input tensor
    """
    for i in range(6):
        minimum = tensor[:, :, i].min()
        maximum = tensor[:, :, i].max()
        tensor[:, :, i] = (tensor[:, :, i] - minimum) / (maximum - minimum)

    return tensor 


def dataloader(path_in, path_out,alpha,beta):
    """
    This function allows us to create denoided and enhanced images from raw images and store 
    them in local folder
    """
    file_names = os.listdir(path_in)
    for file in file_names :
        img = cv2.imread(path_in + file)
        img=cv2.resize(img,(384,384),interpolation = cv2.INTER_AREA)
        print(path_in + file)
        denoised = nlm_denoising(img)
        enhanced = clahe_dwt(denoised,alpha,beta)
        cv2.imwrite(path_out + file, enhanced)
        print(path_out+file)
    return 'END'

def prepare_data(image):
    """
    This is the tranformation function that will be used when creating train and validation loaders
    """
    ## Create the suggested 6 channels 
    tensor = create_channels(image)
    tensor = normalise(tensor)
    return torch.tensor(tensor) 



def train(model, train_loader, val_loader, optimizer, criterion, n_epochs,fold,dir,plotting = True):
    """
    Train the model and plot loss and accuracy plots according to both training and validation
    data
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Check if the folder exists, if not create it
    if not os.path.exists('models/'):
        os.makedirs('models/') 

    model.to(device)    

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_model = copy.deepcopy(model) 
    best_acc = 0 


    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets.data).cpu().sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets.data).cpu().sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)


        if val_acc > best_acc : 
            best_acc = val_acc
            best_model = copy.deepcopy(model)

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}%, Val Loss: {:.4f}, Val Acc: {:.2f}%'
              .format(epoch+1, n_epochs, train_loss, train_acc, val_loss, val_acc))
    
        ## save the model after each fold
        path = dir+f'mymodel-fold-{fold}.pt'
        torch.save(best_model.state_dict(), path)  # saving just weights*


    if plotting : 
        # Plotting
        epochs = range(1, n_epochs+1)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train')
        plt.plot(epochs, val_losses, label='Validation')
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, label='Train')
        plt.plot(epochs, val_accs, label='Validation')
        plt.title('Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
    return best_model
    




# def evaluate(model, test_loader):
#     """
#     Evaluate the model by displaying accuracy and confusion matrix
#     """

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()

#     y_true = []
#     y_pred = []
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             y_true.extend(labels.cpu().numpy())
#             y_pred.extend(predicted.cpu().numpy())
#     cm = confusion_matrix(y_true, y_pred)
#     accuracy = (cm.trace()) / len(y_true)

#     return accuracy, cm



def eval(model, val_loader):
    """
    Evaluate the model in a one vs all approach 
    """

    y_predf = []
    y_truef = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()


    # Define the number of classes
    num_classes = 7

    bs = val_loader.batch_size 

    y_pred = np.empty((bs,num_classes))
    y_true = np.empty((bs,num_classes))

    # Initialize arrays to store the results
    accuracy = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)
    npv = np.zeros(num_classes)
    threat_score = np.zeros(num_classes)


    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)


            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)

            y_truef.extend(labels.cpu().numpy())
            y_predf.extend(predicted.cpu().numpy())
            

            predicted = torch.nn.functional.one_hot(predicted, num_classes=7).float()
            # print('predicted ', predicted)
            # preds = outputs  > threshold
            # preds = preds.float()
            preds = predicted
            # labels = F.one_hot(labels, num_classes=7)
            labels = torch.nn.functional.one_hot(labels, num_classes=7).float()

            # print('labels ' , labels)
            # print('preds ', preds)

            y_true = np.concatenate([y_true,labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, preds.cpu().numpy()])


    y_pred = y_pred[bs:]
    y_true = y_true[bs:]


    # Calculate the confusion matrix for each class
    for i in range(num_classes):

        cm = confusion_matrix(y_true[:, i], 
                                y_pred[:, i], 
                                labels=[0, 1])
        # print('confusion matrix', cm)
        # print('cm ravel', cm.ravel())
        tn, fp, fn, tp = np.array(cm.ravel())

        # Calculate the metrics for each class
        accuracy[i] += (tp + tn) / (tp + tn + fp + fn)
        precision[i] += tp / (tp + fp)
        sensitivity[i] += tp / (tp + fn)
        specificity[i] += tn / (tn + fp)
        npv[i] += tn / (tn + fn)
        threat_score[i] += tp / (tp+fn+fp)

    # # Calculate the average metrics for all classes
    # accuracy = 100 * (accuracy / len(val_loader))
    # precision = 100 *(precision / len(val_loader))
    # sensitivity = 100 * (sensitivity / len(val_loader))
    # specificity = 100 * (specificity / len(val_loader))
    # npv = 100 * (npv /len(val_loader))
    # threat_score = 100 *(threat_score /len(val_loader))

    ## compute accuracy for all classes
    cm = confusion_matrix(y_truef, y_predf)
    fullacc = 100 * (cm.trace()) / len(y_true)
    
    
    print('All classes accuracy : {:.2f}%'.format(fullacc))

    for i in range(num_classes) : 
        print(f'Metrics for class {i}')
        print('Accuracy: {:.2f}%, Precision: {:.2f}%, Sensitivity: {:.2f}%, Specifity: {:.2f}%, NPV: {:.2f}%, Threat score: {:.2f}% '
              .format(accuracy[i], precision[i], sensitivity[i], specificity[i], npv[i], threat_score[i]))
        
    return {'fullacc': fullacc,'accuracy' : accuracy, 'precision' : precision, 'sensitivity' : sensitivity,
            'specificity' : specificity, 'npv' : npv, 'threat_score' : threat_score}

def kfold_validation(model, dataset, optim, criterion ,n_ep , plot, bs, n_sp = 5, rs = 68512) :

    dir = 'models/'+'e'+str(n_ep)+'-bs'+str(bs)+'-'

    best_model = copy.deepcopy(model.state_dict()) 
    best_acc = 0 

    #n_repeats=1
    kf = RepeatedKFold(n_splits= n_sp, random_state=rs)
    accuracy_list = list()

    for fold, (train_ids, val_ids) in enumerate(kf.split(dataset)):
        print(f'FOLD {fold}')
    
        dataset_sizes = {'train': len(train_ids), 'val': len(val_ids)}
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)

        
        train_loader = DataLoader(dataset, batch_size =bs , sampler=train_sampler, num_workers=2, drop_last = True)
        val_loader = DataLoader(dataset, batch_size=bs, sampler=val_sampler, num_workers=2, drop_last= True)

        model_out = train(model, train_loader, val_loader, optim, criterion,plotting=plot,n_epochs = n_ep,fold=fold, dir = dir)
        # accuracy , cm = evaluate(model_out,val_loader)
        # accuracy = 100  * accuracy
        # print(f'Accuracy for fold {fold} is {accuracy:.3f}%')
        # accuracy_list.append(accuracy)

        ## display evaluation plots after each fold
        model_eval = eval(model_out, val_loader)
        accuracy = model_eval['fullacc']
        accuracy_list.append(accuracy)

        # # Check if the folder exists, if not create it
        # if not os.path.exists('models/'):
        #     os.makedirs('models/') 

        # ## save the model after each fold
        # path = dir+f'mymodel-fold-{fold}.pt'
        # torch.save(model.state_dict(), path)  # saving just weights*
    
        if accuracy > best_acc : 
            best_acc = accuracy
            best_model = copy.deepcopy(model.state_dict())

    print(f'Average accuracy of Folds {fold} is {sum(accuracy_list) / len(accuracy_list):.3f}%')  

    # Check if the folder exists, if not create it
    if not os.path.exists('models/best_model/'):
        os.makedirs('models/best_model/')     
    torch.save(best_model, 'models/best_model')
    return accuracy_list
