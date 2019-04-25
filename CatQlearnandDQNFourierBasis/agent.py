from config import config
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from exprience import Exprience
from model import BasicQNet
from model import DistributionNet
from screenManipulation import get_screen

###
import csv
import math
from model import constructTrans
###

class DQNAgent():
    
    #def __init__(self, task, TargetNet, TrainingNet, Test = False, gpu = 0):
    def __init__(self, task, numSmoothTests, Test = False, gpu = 0,catFlag=0):
        
        ###
        n_actions = task.action_space.n
        n_features = len((task.reset()))
        self.cfg=config(n_features, 64, n_actions, n_actions, 51, 200, 0)
        self.cfg.basisNum=4
        
        if catFlag==0:
            TargetNet = BasicQNet(self.cfg)
            TrainingNet = BasicQNet(self.cfg)
            SmoothNet = BasicQNet(self.cfg)
        else:
            TargetNet = DistributionNet(self.cfg)
            TrainingNet = DistributionNet(self.cfg)
            SmoothNet = DistributionNet(self.cfg)
        
        self.catFlag=catFlag
        
        self.cfg.trans = constructTrans(self.cfg)
        ###
        
        self.memory_size = 500000
        self.cfg.batch_size = 128   ## Size of replay batches. Look at one batch per real transition sampled.
        self.lr = 0.00025  ## Parameter that is passed to the Adam optimizer

        self.epsilon = 0.9 #greedy policy epsilon. Parameters are inverted. So epsilon gives prob of being greedy.
        self.epsilon_max = 0.9  ## Caps how greedy we can be. After we reach this we don't decay epsilon anymore
        self.explore_step = 1000000
        
        self.train_start = 0   ### While self.total_step < this we always behave randomly sef.total_step is updated once per action taken
        self.epsilon_decay = (self.epsilon_max - self.epsilon ) / (self.explore_step-self.train_start)
        self.gamma = 1 #discount coefficient
        self.target_replace_iter = 10 ## We set target new = training net every self.target_replace_iter iterations
        
        # self.train_start = 100

        self.storage = Exprience(self.memory_size, self.cfg.batch_size)
        
        self.target_net = TargetNet#.cuda()#.cuda(self.gpu_idx)
        self.training_net = TrainingNet#.cuda()#(self.gpu_idx)
        ###
        self.smoothNet = SmoothNet
        self.numSmoothTests = numSmoothTests
        ###

        self.task = task
        self.test = Test  ### Are we running tests or should we train the network?
        self.total_step = 0
        self.update_count = 1

        self.gpu_idx = gpu
        if not self.test:
            self.optim = torch.optim.Adam(self.training_net.parameters(), lr=self.lr, eps = 0.01/32)
            ###
            self.optimSmooth = torch.optim.Adam(self.smoothNet.parameters(), lr=self.lr, eps = 0.01/32)
            ###
        ###
        if Test and catFlag==0:
            self.training_net.load_state_dict( torch.load("./DQN_model_test.pth",map_location='cpu') )
            self.target_net.load_state_dict( torch.load("./DQN_model_test.pth",map_location='cpu'))
        ###



    def load(self, path, resume = False):
        if not self.test:
            self.target._net = torch.load(path)
        self.training_net = torch.load(path)

    def getValue(self, state, to_np = True):
        state = Variable(torch.from_numpy(state).unsqueeze(0))
        
        ###
        #state = state.unsqueeze(0)
        ###
        
        value = self.training_net(state,self.cfg,0)
        if to_np:
            return value.data.cpu().numpy()
        else:
            return value
    
    def getAction(self, value):
        if self.test:
            #print("problem1")
            return np.argmax(value)
        elif self.total_step < self.train_start:
            #print("problem2")
            return (np.random.randint(0, value.shape[1]))
        elif np.random.rand() < self.epsilon:
            #print("problem3")
            action = np.argmax(value, axis = 1)
            return np.argmax(value, axis = 1)[0] ## This used to be commented out
            return action
        else:
            #print("problem4")
            return np.random.randint(0,value.shape[1])

    def getLoss(self, states, actions, next_states, rewards, status,smoothFlag=0): ## added flag
        states = Variable(torch.Tensor(states))#.cuda()
        actions = Variable(torch.LongTensor(actions))#.cuda()
        rewards = Variable(torch.Tensor(rewards))
        next_states = Variable(torch.Tensor(next_states))#.cuda()
        status = Variable(torch.Tensor(status))
        ###
        if(smoothFlag==0):
            q_train = self.training_net.forward(states,self.cfg,1) # already here
        else:
            q_train = self.smoothNet.forward(states,self.cfg,1)
        ###
        q_train = q_train.gather(1,actions.unsqueeze(1))
        q_target = self.target_net.forward(next_states,self.cfg,1).detach()
        q_target = rewards.unsqueeze(1) + (1 - status.unsqueeze(1))*(self.gamma * q_target.max(1)[0]).view(-1, 1) # shape (batch, 1)
        
        loss = F.mse_loss(q_train, q_target)
        return loss

    def episode(self):
        c_state = self.task.reset()
        
        ####
        #self.task.reset()
        #last_screen = get_screen(self.task)
        #current_screen = get_screen(self.task)
        #state = current_screen - last_screen
        #state = state.view((-1,))
        ###
        
        done = False
        total_reward = 0
        episode_step = 0
        while not done:
            
            ###
            if episode_step >=200:
                break
            ###
            c_state = c_state.astype(np.float32)
            value = self.getValue(c_state)
            ### commented out when got downloaded: self.task.render()
            
            ###
            #self.task.render()
            ###
            
            ###
            #value = self.getValue(state)
            ###
            
            action = self.getAction(value)
            next_state, reward, done, _ = self.task.step(action)
        
            ###
            #_, reward, done, _ = self.task.step(action)
            #last_screen = current_screen
            #current_screen = get_screen(self.task)
            #next_state = current_screen - last_screen
            #next_state = next_state.view((-1,))
            ###
            
            total_reward += reward

            if not self.test:
                self.storage.store_experience((c_state, action, next_state, reward, float(done)))
                #self.storage.store_experience((state, action, next_state, reward,float(done)))
                self.total_step += 1


            c_state = next_state
            
            ###
            #state = next_state
            ###
            
            if done: 
                break
            
            if not self.test and self.total_step >= self.train_start:
                if self.epsilon < self.epsilon_max:  # Decay epsilon
                    self.epsilon += self.epsilon_decay
                experiences = self.storage.resample()
                states, actions, next_states, rewards, status = experiences
                    
                self.optim.zero_grad()
                loss = self.getLoss(states, actions, next_states, rewards, status)
                loss.backward()
                self.optim.step()

                ###
                for i in range(self.numSmoothTests):
                    self.smoothNet.load_state_dict(self.training_net.state_dict())
                    with torch.no_grad():
                        for param in self.smoothNet.parameters():
                            param.add_(torch.randn(param.size()) * 0.1)
                    self.optimSmooth.zero_grad()
                    loss = self.getLoss(states, actions, next_states, rewards, status,smoothFlag=1)
                    loss.backward()
                    
                    totalNormGrad=0
                    totalNormWeight=0
                    count1=0
                    for p in self.smoothNet.parameters():
                        count2=0
                        count1=count1+1
                        for q in self.training_net.parameters():
                            count2=count2+1
                            if count2==count1:
                                totalNormGrad += (p.grad.data-q.grad.data).norm(2) ** 2
                                totalNormWeight += (p.data-q.data).norm(2) ** 2

                    totalNormGrad = totalNormGrad ** (0.5)
                    totalNormWeight = totalNormWeight ** (0.5)
                
                    if(self.catFlag==0):
                        with open('DQNSmooth.csv','a') as writeFile:
                            writer = csv.writer(writeFile)
                            writer.writerow(np.array([totalNormWeight/totalNormGrad,self.total_step]))
                    else:
                        with open('CatSmooth.csv','a') as writeFile:
                            writer = csv.writer(writeFile)
                            writer.writerow(np.array([totalNormWeight/totalNormGrad,self.total_step]))
                ###


                ###
                totalNorm=0
                for p in self.training_net.parameters():
                    totalNorm += p.grad.data.norm(2) ** 2
                totalNorm = totalNorm ** (0.5)
                if self.catFlag==0:
                    with open('DQNNorms.csv','a') as writeFile:
                        writer = csv.writer(writeFile)
                        writer.writerow(np.array([totalNorm,self.total_step]))
                else:
                    with open('CategoricalNorms.csv','a') as writeFile:
                        writer = csv.writer(writeFile)
                        writer.writerow(np.array([totalNorm,self.total_step]))
                ###

                    
                    
            if not self.test and self.total_step % self.target_replace_iter == 0:
                self.target_net.load_state_dict(self.training_net.state_dict())
    
            ###
            episode_step = episode_step + 1
            ###
        return total_reward


class CategoricalDQNAgent(DQNAgent):
    def __init__(self, task,numSmoothTests, Test = False, gpu = 0):
        super(CategoricalDQNAgent, self).__init__(task,numSmoothTests, Test, gpu, catFlag=1)
        self.atoms = torch.from_numpy(np.linspace(self.cfg.categorical_v_min,
                        self.cfg.categorical_v_max,
                        self.cfg.atomNum)).float()
        self.delta_atom = (self.cfg.categorical_v_max - self.cfg.categorical_v_min) / float(self.cfg.atomNum - 1)
        ###
        if Test:
            self.training_net.load_state_dict( torch.load("./Categorical_DQN_model_test.pth",map_location='cpu') )
            self.target_net.load_state_dict( torch.load("./Categorical_DQN_model_test.pth",map_location='cpu'))



    def getValue(self, state, to_np = True):
        state = Variable(torch.from_numpy(state).unsqueeze(0))#.cuda()
        
        ###
        #state = state.unsqueeze(0)
        ###
        
        distrib = self.training_net(state,self.cfg,0).detach().data.cpu()
        
        #value = ( distrib * self.atoms ).sum(dim = 1)
        
        ###
        value = ( distrib * self.atoms ).sum(dim=1).unsqueeze(0)
        ###
        
        if to_np:
            return value.cpu().numpy()
        else:
            return value
        
        
    def getLoss(self, states, actions, next_states, rewards, status,smoothFlag=0):
        states = Variable(torch.Tensor(states))#.cuda()
        actions = Variable(torch.LongTensor(actions))#.cuda()
        rewards = Variable(torch.Tensor(rewards))
        next_states = Variable(torch.Tensor(next_states))#.cuda()
        status = Variable(torch.Tensor(status))

        #training distribution
        if smoothFlag==0:
            distrib_train = self.training_net(states,self.cfg,1)
        else:
            distrib_train = self.smoothNet(states,self.cfg,1)
        actions = (actions.unsqueeze(1)).repeat(1,self.cfg.atomNum).unsqueeze(1)
        distrib_train = distrib_train.gather(1,actions).squeeze(1)

        #next action distribution
        distrib_next = self.target_net.forward(next_states,self.cfg,1)#.detach().data.cpu()
        #Q_next = (distrib_next * self.atoms).sum(-1)
        #_, best_actions = Q_next.max(dim = 1)
        ###
        Q_next = (distrib_next * self.atoms).sum(dim=2)
        ###
        #_ , best_actions = Q_next.max(dim = 1)
        ###
        _ , best_actions = Q_next.max(dim = 1)
        ###
        best_actions = best_actions.unsqueeze(1).repeat(1,self.cfg.atomNum).unsqueeze(1)
        distrib_next = distrib_next.gather(1, best_actions).squeeze(1) #BatchSize * AtomNum
        
        #shift atoms
        shifted_atoms = (1-status).view(-1,1) * self.atoms.view(1,-1) * self.gamma + rewards.view(-1,1) #BatchSize * AtomNum
        shifted_atoms = shifted_atoms.clamp(self.cfg.categorical_v_min, self.cfg.categorical_v_max)

        b = (shifted_atoms - self.cfg.categorical_v_min)/self.delta_atom
        l = b.floor()
        u = b.ceil()
        
        d_m_l = (u + (l == u).float() - b) * distrib_next
        d_m_u = (b - l) * distrib_next
        
        # target_prob = self.learning_network.tensor(np.zeros(prob_next.size()))
        distrib_target = torch.Tensor(torch.from_numpy(np.zeros( (self.cfg.batch_size, self.cfg.atomNum)) ).float() )

        for i in range(self.cfg.batch_size):
            distrib_target[i].index_add_(0, l[i].long(), d_m_l[i])
            distrib_target[i].index_add_(0, u[i].long(), d_m_u[i])
        
        
        # loss = F.binary_cross_entropy(distrib_train, Variable(distrib_target).cuda())   ## COMENTED OUT WHEN I GOT IT
        #loss = -(Variable(distrib_target).cuda() * distrib_train.log()).sum(-1).mean()   ## COMMENTED OUT TO REMOVE CUDA
        
        #### KL divergence as the loss
        loss = -(Variable(distrib_target).detach() * distrib_train.log()).sum(-1).mean()
        
        #### Cramer loss
        
        #cdfMask = torch.Tensor(torch.from_numpy(np.zeros( (self.cfg.atomNum, self.cfg.atomNum)) ).float() )
        #for i in range(self.cfg.atomNum):
        #    for j in range(i+1):
        #        cdfMask[j,i] = 1

#loss = ((torch.mm(distrib_target.detach() , cdfMask) - torch.mm(distrib_train , cdfMask)).pow(2)*self.delta_atom).sum(1).pow(0.5).mean()
        ###
        
        return loss

class CategoricalDQNAgent_Pong(CategoricalDQNAgent):
    def __init__(self, task, TargetNet, TrainingNet, Test = False, gpu = 0):
        super(CategoricalDQNAgent_Pong, self).__init__(task, TargetNet, TrainingNet, Test, gpu)
        self.temp_experience = []

    def episode(self):
        c_state = self.task.reset()
        done = False
        total_reward = 0
        episode_step = 0
        while not done:
            c_state = c_state.astype(np.float32)
            value = self.getValue(c_state)
            # self.task.render()

            action = self.getAction(value)
            next_state, reward, done, _ = self.task.step(action)
        
            total_reward += reward

            # if not self.test:
            #     self.storage.store_experience((c_state, action, next_state, reward,float(done)))
            #     self.total_step += 1

            if not self.test:
                if reward != 0:
                    t_reward = reward
                    for exp in self.temp_experience:
                        s, a, n_s, _, sta = exp
                        self.storage.store_experience((s, t_reward, n_s, a, sta ))
                        t_reward *= self.gamma
                        # print (t_reward)
                        self.total_step += 1
                    # del self.temp_experience
                    self.temp_experience = []
                else:
                    self.temp_experience.append((c_state, action, next_state, reward,float(done)))

            c_state = next_state

            if done: 
                break
            
            if not self.test and self.total_step >= self.train_start:
                if self.epsilon < self.epsilon_max:
                    self.epsilon += self.epsilon_decay
                experiences = self.storage.resample()
                states, actions, next_states, rewards, status = experiences
                # print (states.shape)
                # print (actions.shape)
                # print (next_states.shape)
                # print (rewards.shape)
                # print (status.shape)

                self.optim.zero_grad()
                loss = self.getLoss(states, actions, next_states, rewards, status)
                loss.backward()
                self.optim.step()
            
            if not self.test and self.total_step % self.target_replace_iter == 0:
                self.target_net.load_state_dict(self.training_net.state_dict())
        
        return total_reward
            
        c_state = self.task.reset()
        done = False
        total_reward = 0
        while not done:
            c_state = c_state.astype(np.float32)
            value = self.getValue(c_state)
            # self.task.render()
            action = self.getAction(value)
            next_state, reward, done, _ = self.task.step(action)
        
            total_reward += reward

            if not self.test:
                if reward != 0:
                    t_reward = reward
                    for exp in self.temp_experience:
                        s, a, n_s, _, sta = exp
                        self.storage.store_experience((s, t_reward, n_s, a, sta ))
                        t_reward *= self.gamma
                        # print (t_reward)
                        self.total_step += 1
                    # del self.temp_experience
                    self.temp_experience = []
                else:
                    self.temp_experience.append((c_state, reward, next_state, action,float(done)))

            c_state = next_state

            if done:
                break
            
            if not self.test and self.total_step >= self.train_start:
                if self.epsilon < self.epsilon_max:
                    self.epsilon += self.epsilon_decay
                experiences = self.storage.resample()
                states, actions, next_states, rewards, status = experiences

                self.optim.zero_grad()
                loss = self.getLoss(states, actions, rewards, next_states,status)
                # print(loss)
                loss.backward()
                self.optim.step()
            
            if not self.test and self.total_step % self.target_replace_iter == 0:
                self.target_net.load_state_dict(self.training_net.state_dict())
        
        return total_reward
