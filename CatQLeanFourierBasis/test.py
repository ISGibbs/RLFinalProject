import torch
import gym
from task import PongTask
from agent import DQNAgent
from agent import CategoricalDQNAgent
from agent import CategoricalDQNAgent_Pong
from model import BasicQNet
from model import DistributionNet
from model import CategoricalConvNet

#from config import cfg


def test_categorical(seed, numEpisodes):
    task = gym.make('CartPole-v0').unwrapped
    task.reset()
    task.seed(seed)
    #Net = DistributionNet(4, cfg.actionNum,51)
    #Net2 = DistributionNet(4, cfg.actionNum,51)
    
    #Net.load_state_dict( torch.load("./Categorical_DQN_model.pth",map_location='cpu') )
    #Net2.load_state_dict( torch.load("./Categorical_DQN_model.pth",map_location='cpu'))
    
    # Net.load_state_dict(.state_dict()) Already commented out
    
    agent = CategoricalDQNAgent(task,0, Test = True, gpu = 0)
    ep = 0
        #while True:
        #ep += 1
        #r = agent.episode()
#print ('Ep:{} Reward {}'.format(ep,r))
    average_reward=0
    for i in range(numEpisodes):
        average_reward =average_reward + agent.episode()

    average_reward=average_reward/numEpisodes

#task.render()
    task.close()
    return average_reward

def testDQN(seed, numEpisodes):
    task = gym.make('CartPole-v0').unwrapped
    task.reset()
    task.seed(seed)
    
    agent = DQNAgent(task,0, Test = True, gpu = 0,catFlag=0)

    ep = 0
    average_reward=0
    for i in range(numEpisodes):
        average_reward = average_reward + agent.episode()

    average_reward=average_reward/numEpisodes

#task.render()
    task.close()
    return average_reward


if __name__ == "__main__":
    # train()
    test_categorical()
    # train_categorical_Pong()
