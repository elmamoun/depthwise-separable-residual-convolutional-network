import torch.nn as nn 
from torch.nn import init

class ConvBlock(nn.Module):
    """
    The class defines the Convolution Block defined in the paper's architecture
    """
    def __init__(self, activation= nn.ReLU(), cin = 6, cout = 16 ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv3d(cin, cin, kernel_size=(1,4,4), stride=1, dilation =(1,2,2), padding=(0,1,1),
                               groups = cin),
                    nn.BatchNorm3d(cin),
                    nn.Conv3d(cin, cout, kernel_size=(1,1,1), stride=1, padding=(0,2,2)),
                    nn.BatchNorm3d(cout),
                    activation)
        
        init.kaiming_uniform_(self.conv[0].weight, nonlinearity='linear')
        if activation == nn.ReLU():
            init.kaiming_uniform_(self.conv[2].weight, nonlinearity='relu')
        elif activation == nn.ELU() : 
            init.kaiming_uniform_(self.conv[2].weight, nonlinearity='elu')
        elif activation == nn.Identity(): 
            init.kaiming_uniform_(self.conv[2].weight, nonlinearity='linear')
            

        
    
    # forward function
    def forward(self,x) :
        x = self.conv(x)
        return x 


class ResBlock(nn.Module):
    """
    This class defines the Residual module defined in the paper's architecture
    """
    def __init__(self,cin1,cout1,cin2,cout2,cin3,cout3,cin4,cout4):
        super().__init__()
        self.conv1 = ConvBlock(cin=cin1, cout = cout1 ,activation=nn.ELU())
        self.conv2 = ConvBlock(cin = cin2, cout = cout2, activation=nn.ELU())
        self.conv3 = ConvBlock(cin = cin3, cout = cout3, activation=nn.Identity())
        self.conv4 = ConvBlock(cin= cin4, cout= cout4, activation= nn.Identity())


    # upper block of residual module composed of 3 successive convolutions
    def upper_block(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    # lower block of residual module composed of single convolution 
    def lower_block(self, x): 
        return self.conv4(x)
    # forward function
    def forward(self,x): 
        x = self.lower_block(x) + self.upper_block(x)
        return x 


class MyModel(nn.Module):
    """
    Defining our final model as presented in the research paper     
    """
    def __init__(self):
        super(MyModel, self).__init__()

        ## Add the first convolution block
        self.conv1 = ConvBlock()

        ## Add the second convolution block
        self.conv2 = ConvBlock(cin=32, cout=32)

        ## Add the third convolution block
        self.conv3 = ConvBlock(cin=48, cout=48)

        ## Add the fourth convolution block
        self.conv4 = ConvBlock(cin=64, cout=64)

        ## Add the fifth convolution block
        self.conv5 = ConvBlock(cin=96, cout=96)

        ## Maxpooling
        self.mp = nn.MaxPool3d(kernel_size=(1, 10, 10), stride=(1, 2, 2), padding=(0,5,5))

        ## Add the first residual block
        self.res1 = ResBlock(cin1 = 16,cout1 = 16,
                            cin2 = 16,cout2 = 16,
                            cin3 = 16,cout3 = 32,
                            cin4 = 16,cout4 = 32)
        
        ## Add the second residual block
        self.res2 = ResBlock(cin1 = 32,cout1 = 32,
                            cin2 = 32,cout2 = 32,
                            cin3 = 32,cout3 = 48,
                            cin4 = 32,cout4 = 48)
        
        ## Add the third residual block 
        self.res3 = ResBlock(cin1 = 48,cout1 = 48,
                            cin2 = 48,cout2 = 48,
                            cin3 = 48,cout3 = 64,
                            cin4 = 48,cout4 = 64)
        
        ## Add the fourth residual block
        self.res4 = ResBlock(cin1 = 64,cout1 = 64,
                            cin2 = 64,cout2 = 64,
                            cin3 = 64,cout3 = 96,
                            cin4 = 64,cout4 = 96)

        ## Add the GAP layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        ## Add the denser layer
        # self.fc = nn.Linear(96, 512)
        self.fc = nn.Sequential(
            nn.Linear(96,512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        init.kaiming_uniform_(self.fc[0].weight, nonlinearity='relu')

        # Add a softmax layer for 7-class classification
        self.softmax = nn.Sequential(
            nn.Linear(512, 7),
            nn.Softmax(dim=1))

    def forward(self, x):
        x = x.unsqueeze(2)           # add depth dimension 
        x = x.permute(0, 4, 2, 1,3)  # change shape to [batch_size, channels,depth, height, width]
        x = self.conv1(x)
        x = self.mp(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.mp(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.mp(x)
        x = self.res3(x)
        x = self.conv4(x)
        x = self.mp(x)
        x = self.res4(x)
        x = self.conv5(x)
        x = self.mp(x)
        x = self.global_avg_pool(x)
        ## flatten the tensor
        x = x.view(-1, 96)
        ## add the fully connected layer
        x = self.fc(x)
        ## add the softmax layer
        x = self.softmax(x)

        return x
    