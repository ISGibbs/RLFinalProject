import numpy as np

class Exprience():
    
    def __init__(self, memory_size, batch_size ):
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.stored_states = None
        self.stored_next_states = None
        self.stored_actions = np.empty(self.memory_size, dtype = np.uint8)
        self.stored_rewards = np.empty(self.memory_size)
        self.stored_status = np.empty(self.memory_size)

        self.expr_pos = 0
        self.expr_full = False
    
    def store_experience(self, experience):

        state, action, next_state, reward, done = experience
        ###
        #state=state.numpy()
        #next_state=next_state.numpy()
        ###

        if self.stored_states is None:
            self.stored_states = np.empty((self.memory_size, ) + state.shape, dtype=state.dtype)
            self.stored_next_states = np.empty((self.memory_size, ) + state.shape, dtype=next_state.dtype)
        
        self.stored_states[self.expr_pos][:] = state
        self.stored_next_states[self.expr_pos][:] = next_state
        self.stored_rewards[self.expr_pos] = reward
        self.stored_actions[self.expr_pos] = action
        self.stored_status[self.expr_pos] = done

        self.expr_pos += 1
        if self.expr_pos >= self.memory_size:
            self.expr_pos = 0   
            self.expr_full = True

    def resample(self):
        total_exprience = self.memory_size if self.expr_full else self.expr_pos
        
        idx = np.random.randint(0, total_exprience, size = self.batch_size)

        return (
            self.stored_states[idx], 
            self.stored_actions[idx],
            self.stored_next_states[idx],
            self.stored_rewards[idx], 
            self.stored_status[idx]
        )
        
