
class config():
    def __init__(self,inputNum, hiddenNum, outputNum, actionNum, atomNum, catMax, catMin):
        # self.inputNum = 6400 * 4  ## Number of input nodes
        # self.hiddenNum = 256  ## Number of hidden nodes
        # self.outputNum = 2   ## Number of output nodes
        self.atomNum = atomNum  ## Parameterizes the number of atoms in the distribution
        self.actionNum = actionNum ## Number of actions in the enviroment
        
        ### For simple cartpole
        self.inputNum=inputNum
        self.hiddenNum=hiddenNum
        self.outputNum=outputNum
        
        #####################
        self.categorical_v_max = catMax ## Value of the maximum atom
        self.categorical_v_min = catMin ## Value of the minimum atom

        self.save_inter = 50  ## How often should we save to model
        # self.model_path = './DQN_model.pth'
        self.model_path = './Categorical_DQN_model.pth'


        self.seed=0

def computeCDF(pdf,cfg):
    cdf_train = torch.Tensor(torch.from_numpy(np.zeros( (cfg.batch_size, cfg.atomNum)) ).float() )
    for i in range(cfg.atomNum):
        cdf_train[:,i] = pdf[:,0:(i+1)].sum(dim=1)
    return cdf_train

