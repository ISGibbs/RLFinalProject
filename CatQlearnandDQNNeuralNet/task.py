import gym
import numpy as np

class BasicTask():

    def __init__(self, t):
        self.env = t
    def step(self,action):
        return self.env.step(action)
    def reset(self):
        return self.env.reset()
    def render(self):
        self.env.render()

class PongTask(BasicTask):
    def __init__(self, t):
        super(PongTask, self).__init__(t)
        self.frame = np.ndarray((4,80,80))

    def prepro(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float)

    def step(self,action):
        next_state, reward, done, info = self.env.step(action)
        # print (next_state.shape)
        next_state = self.prepro(next_state)
        self.frame = np.roll(self.frame, -1, axis = 0)
        self.frame[3] = next_state
        # x = next_state - self.pre_state
        # self.pre_state = next_state
        return self.frame, reward, done, info

    def reset(self):
        temp_state = self.env.reset()
        temp_state = self.prepro(temp_state)
        self.frame[:] = temp_state
        # print (self.frame.shape)
        return self.frame
    # def render(self):
    #     self.env.render()