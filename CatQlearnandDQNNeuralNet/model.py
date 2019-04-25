import torch 
import torch.nn as nn
import torch.nn.functional as F
# from config import cfg



class BasicQNet(nn.Module):

    def __init__(self,cfg):
        super(BasicQNet,self).__init__()
        self.fc1 = nn.Linear(cfg.inputNum, cfg.hiddenNum)   ### one layer linear network that takes input of dimension inputNum and returns output of dimension hiddenNum
        self.fc2 = nn.Linear(cfg.hiddenNum, cfg.hiddenNum)
        self.fc3 = nn.Linear(cfg.hiddenNum, cfg.outputNum)

        self.init_weights()

    def init_weights(self):
        #init all weights using xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)    ## normal initialization with mean 0 and variance = 2/(fan_in + fan_out) where fan_in=#(input nodes to the layer), fan_out=#(output nodes at the layer)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # out = F.softmax(out, dims = 1)
        
        return out


class DistributionNet(nn.Module):

    def __init__(self,cfg):
        super(DistributionNet,self).__init__()
        self.fc1 = nn.Linear(cfg.inputNum, cfg.hiddenNum)
        self.fc2 = nn.Linear(cfg.hiddenNum, cfg.hiddenNum)
        self.fc3 = nn.Linear(cfg.hiddenNum, cfg.actionNum * cfg.atomNum)
        
    
    def init_weights(self):
        #init all weights using xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)        

    def forward(self,x,cfg,batchFlag):

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        if batchFlag==1:
            out=out.view(cfg.batch_size,2,cfg.atomNum)
            out = F.softmax(out, dim = 2)
        else:
            out=out.view(2,cfg.atomNum)
            out = F.softmax(out, dim = 1)
                
        return out

class CategoricalConvNet(nn.Module):
    def __init__(self, in_channels, n_actions, n_atoms, gpu=0):
        super(CategoricalConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(6 * 6 * 64, 512)
        self.fc_categorical = nn.Linear(512, n_actions * n_atoms)
        self.n_actions = n_actions
        self.n_atoms = n_atoms

    def forward(self, x):
        # x = self.variable(x)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        # print(y.shape)
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        y = self.fc_categorical(y)
        y = y.view(-1,self.n_actions,self.n_atoms)
        y = F.softmax(y, dim = 2)
        return y
