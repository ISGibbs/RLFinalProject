import torch 
import torch.nn as nn
import torch.nn.functional as F
# from config import cfg

def constructTrans(cfg):
    count=0
    temp = torch.zeros([((cfg.basisNum+1)**cfg.inputNum),cfg.inputNum])
    i=0
    while i < cfg.basisNum+1:
        temp[count][0] = i
        j=0
        while j < cfg.basisNum+1:
            temp[count][1] = j
            k = 0
            while k < cfg.basisNum+1:
                temp[count][2] = k
                l = 0
                while l < cfg.basisNum+1:
                    temp[count][3] = l
                    count = count + 1
                    if count < (cfg.basisNum+1)**cfg.inputNum:
                        temp[count][0] = i
                        temp[count][1] = j
                        temp[count][2] = k
                    l = l + 1
                k = k + 1
            j = j + 1
        i = i + 1
    return temp


def makeBasis(state,cfg,batchFlag):
    if batchFlag==1:
        basis = torch.zeros([2*((cfg.basisNum+1)**cfg.inputNum),cfg.batch_size])
    else:
        basis = torch.zeros([2*((cfg.basisNum+1)**cfg.inputNum)])
    count=0
    temp = torch.zeros([1,cfg.inputNum])
    i=0
    state = state.transpose(0,1)
    while i < cfg.basisNum+1:
        temp[0][0] = i
        j=0
        while j < cfg.basisNum+1:
            temp[0][1] = j
            k = 0
            while k < cfg.basisNum+1:
                temp[0][2] = k
                l = 0
                while l < cfg.basisNum+1:
                    temp[0][3] = l
                    basis[count] = (torch.mm(temp,state)*3.14159265359).cos()
                    count = count + 1
                    basis[count] = (torch.mm(temp,state)*3.14159265359).sin()
                    count = count + 1
                    l = l + 1
                k = k + 1
            j = j + 1
        i = i + 1
    return basis


class BasicQNet(nn.Module):
    
    def __init__(self,cfg):
        super(BasicQNet,self).__init__()
        self.fc1 = nn.Linear(2*((cfg.basisNum+1)**cfg.inputNum), cfg.outputNum)
        
        self.init_weights()
    
    def init_weights(self):
        #init all weights using xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)    ## normal initialization with mean 0 and variance = 2/(fan_in + fan_out) where fan_in=#(input nodes to the layer), fan_out=#(output nodes at the layer)

    def forward(self,state,cfg, batchFlag):
        #x = makeBasis(state,cfg, batchFlag)
        #if batchFlag==1:
        #    x = x.transpose(0,1)
        #out = self.fc1(x)
        #if batchFlag==0:
        #    out = out.unsqueeze(0)
        # out = F.softmax(out, dims = 1)
        
        x = (torch.mm(cfg.trans,state.transpose(0,1))*3.14159265359).cos()
        y = (torch.mm(cfg.trans,state.transpose(0,1))*3.14159265359).sin()
        x = torch.cat((x, y), 0)
        out = self.fc1(x.transpose(0,1))

        return out


class DistributionNet(nn.Module):

    def __init__(self,cfg):
        super(DistributionNet,self).__init__()
        self.fc1 = nn.Linear(2*((cfg.basisNum+1)**cfg.inputNum), cfg.actionNum * cfg.atomNum)
    
    def init_weights(self):
        #init all weights using xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)        

    def forward(self,state,cfg,batchFlag):
        #x = makeBasis(state,cfg, batchFlag)
        #if batchFlag==1:
        #x = x.transpose(0,1)
                #out = self.fc1(x)
                #if batchFlag==0:
                #out = out.unsqueeze(0)
        x = (torch.mm(cfg.trans,state.transpose(0,1))*3.14159265359).cos()
        y = (torch.mm(cfg.trans,state.transpose(0,1))*3.14159265359).sin()
        x = torch.cat((x, y), 0)
        out = self.fc1(x.transpose(0,1))
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
