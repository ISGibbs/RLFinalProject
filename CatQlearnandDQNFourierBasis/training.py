import torch

import gym
from task import PongTask

from agent import DQNAgent
from agent import CategoricalDQNAgent
from agent import CategoricalDQNAgent_Pong
from model import BasicQNet
from model import DistributionNet
from model import CategoricalConvNet
# from config import cfg

###
import csv
import numpy as np
import random
from test import test_categorical
from test import testDQN
###

def JitterTestDQN(numJitterTest,numTestEpisodes,iteration,training_net):
    with open('jitteredTestRewards.csv','a') as writeFile:
        writer = csv.writer(writeFile)
        
        ### one no jitter test
        tempnet = training_net
        with open("./DQN_model_test.pth", 'wb') as f:
            torch.save(tempnet.state_dict(), f)
        r = testDQN(9999 + iteration,numTestEpisodes)
        writer.writerow(np.array([r]))
        print ('Trial{} Test Number:{} Reward {}'.format(iteration,-1,r))
        
        ### jitter tests
        for j in range(numJitterTest):
            tempnet.load_state_dict(torch.load("./DQN_model.pth",map_location='cpu') )
            with torch.no_grad():
                for param in tempnet.parameters():
                    param.add_(torch.randn(param.size()) * 0.01)
            with open("./DQN_model_test.pth", 'wb') as f:
                torch.save(tempnet.state_dict(), f)
            
            r = testDQN(99999 + iteration + j*100000,numTestEpisodes)
            writer.writerow(np.array([r]))

def JitterTestCat(numJitterTest,numTestEpisodes,iteration,training_net):
    with open('jitteredCategoricalTestRewards.csv','a') as writeFile:
        writer = csv.writer(writeFile)
        
        ### one no jitter test
        tempnet = training_net
        with open("./Categorical_DQN_model_test.pth", 'wb') as f:
            torch.save(tempnet.state_dict(), f)
        r = test_categorical(9999 + iteration,numTestEpisodes)
        writer.writerow(np.array([r]))
        print ('Trial{} Test Number:{} Reward {}'.format(iteration,-1,r))
        
        ### jitter tests
        for j in range(numJitterTest):
            tempnet.load_state_dict(torch.load("./Categorical_DQN_model.pth",map_location='cpu') )
            with torch.no_grad():
                for param in tempnet.parameters():
                    param.add_(torch.randn(param.size()) * 0.01)
            with open("./Categorical_DQN_model_test.pth", 'wb') as f:
                torch.save(tempnet.state_dict(), f)
            
            r = test_categorical(99999 + iteration+j*100000,numTestEpisodes)
            writer.writerow(np.array([r]))


def train(seed, numEpisodes,save_iterations,numJitterTest,numTestEpisodes,iteration,numSmoothTests):
    #task = gym.make("CartPole-v0")
    ###
    task = gym.make('CartPole-v0').unwrapped
    task.reset()
    task.seed(seed)
    ###
   
    #agent = DQNAgent(task, BasicQNet(), BasicQNet(),Test = False, gpu = 0)
    ###
    save_path = './DQN_model.pth'
    agent = DQNAgent(task,numSmoothTests,Test = False, gpu = 0,catFlag=0)
    ###

    ep = 0
    runing_avg = -21
    ###
    with open('DQNRewards.csv','a') as writeFile:
        writer = csv.writer(writeFile)
        ###
        for i in range(numEpisodes):
            ep += 1
            r = agent.episode()
            ###
            writer.writerow(np.array([r]))
            ###
            print ('Ep:{} Reward {}'.format(ep,r))
            if ep % save_iterations == 0:
                with open(save_path, 'wb') as f:
                    torch.save(agent.training_net.state_dict(), f)
                JitterTestDQN(numJitterTest,numTestEpisodes,iteration,agent.training_net)

    with open(save_path, 'wb') as f:
        torch.save(agent.training_net.state_dict(), f)
    ###
    task.close()
    return agent.training_net
    ###

def train_categorical(seed,numEpisodes,save_iterations,numJitterTest,numTestEpisodes,iteration,numSmoothTests):
    #task = gym.make("CartPole-v0")
    ###
    task = gym.make('CartPole-v0').unwrapped
    task.reset()
    task.seed(seed)
    ###
    
    #agent = DQNAgent(task, BasicQNet(), BasicQNet(),Test = False, gpu = 0)
    ###
    save_path = './Categorical_DQN_model.pth'
    agent = CategoricalDQNAgent(task,numSmoothTests,Test = False, gpu = 0)
    ###
    
    ep = 0
    runing_avg = -21
    ###
    with open('categoricalRewards.csv','a') as writeFile:
        writer = csv.writer(writeFile)
        ###
        for i in range(numEpisodes):
            ep += 1
            r = agent.episode()
            ###
            writer.writerow(np.array([r]))
            ###
            print ('Ep:{} Reward {}'.format(ep,r))
            if ep % save_iterations == 0:
                with open(save_path, 'wb') as f:
                    torch.save(agent.training_net.state_dict(), f)
                JitterTestCat(numJitterTest,numTestEpisodes,iteration,agent.training_net)
    
        with open(save_path, 'wb') as f:
            torch.save(agent.training_net.state_dict(), f)
        
    ###
    task.close()
    return agent.training_net
    ###


def main_cat(numTrainEpisodes, numJitterTest, numTestEpisodes,numRepititions,save_iterations,numSmoothTests):
    for i in range(numRepititions):
        print ('ITERATION:{} STARTED______________________________________________________________'.format(i))
        np.random.seed(111311+i)  ## 0 = seed
        torch.manual_seed(11233334+i)
        #train(seed=44444+i)
        net = train_categorical(11130+i,numTrainEpisodes,save_iterations,numJitterTest,numTestEpisodes,i,numSmoothTests)
#print ('Test Number:{} Reward {}'.format(j,r))

def main_DQN(numTrainEpisodes, numJitterTest, numTestEpisodes,numRepititions,save_iterations,numSmoothTests):
    for i in range(numRepititions):
        print ('ITERATION:{} STARTED______________________________________________________________'.format(i))
        np.random.seed(111311+i)  ## 0 = seed
        torch.manual_seed(11233334+i)
        #train(seed=44444+i)
        net = train(11130+i,numTrainEpisodes,save_iterations,numJitterTest,numTestEpisodes,i,numSmoothTests)
#print ('Test Number:{} Reward {}'.format(j,r))


if __name__ == "__main__":
    main_DQN(numTrainEpisodes=100,numJitterTest=1,numTestEpisodes=1,numRepititions=32,save_iterations=10,numSmoothTests=2)
    main_cat(numTrainEpisodes=200,numJitterTest=1,numTestEpisodes=1,numRepititions=32,save_iterations=10,numSmoothTests=2)




