import random


class GameAgent:
    '''
    Base class for game agents
    '''

    def train(self, game_state):
        '''
        Run a step of training
        '''
        return True


    def get_action(self, state):
        '''
        Get action for given state
        '''
        raise NotImplementedError


    def is_training_done(self):
        '''
        Return whether training is done
        '''
        return True


    def open(self):
        '''
        Make agent ready for use
        '''
        pass


    def close(self, ty=None, value=None, tb=None):
        '''
        Close down agent
        '''
        pass


    def __enter__(self):
        '''
        Allow for with statement
        '''
        return self.open()


    def __exit__(self, ty, value, tb):
        '''
        Allow for with statement
        '''
        self.close(ty, value, tb)


class RandomAgent(GameAgent):
    '''
    Does random actions
    '''

    def __init__(self, conf):
        self.action_space = conf['action_space']

    
    def get_action(self, state):
        return random.randint(0, self.action_space - 1)
