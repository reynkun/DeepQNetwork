import os
import importlib
import gym
import time
import tensorflow as tf


from ..utils.logging import log


class Model:
    '''
    Model handles the complexities of the tensorflow setup and tear down.
    Also adds model (auto) saving and loading
    '''

    DEFAULT_OPTIONS = {
        'save_path_prefix': None
    }

    def __init__(self, 
                 conf, 
                 init_model=False, 
                 load_model=True,
                 save_model=False):
        self.conf = conf
        self.init_model = init_model
        self.load_model = load_model
        self.save_model = save_model


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


    def run(self, *args, **kwargs):
        '''
        Runs a tf operation
        '''
        return self._sess.run(*args, **kwargs)


    def save(self, save_path_prefix):
        '''
        Saves the model
        '''

        log('saving model: ', save_path_prefix)
        self.saver.save(self._sess, save_path_prefix)
        log('saved model')


    def restore(self, save_path_prefix):
        '''
        Restores the model
        '''

        if not os.path.exists(save_path_prefix + '.index'):
            log('  model does not exist:', save_path_prefix)
            return False

        log('  restoring model: ', save_path_prefix)
        self.saver.restore(self._sess, save_path_prefix)
        log('  restored model')

        return True


    def open(self):
        '''
        Initializes the model
        '''
        log('creating new session load_model:', self.load_model, 'save_model:', self.save_model)

        tf.reset_default_graph()

        if self.init_model:
            self.make_model()

        self._init_tf()

        return self


    def make_model():
        '''
        Makes the actual model.  Subclass should override
        '''
        raise NotImplementedError


    def _init_tf(self):
        '''
        Runs tensorflow specific setup
        '''
        # make saver
        self.saver = tf.train.Saver()

        # set tensorflow to NOT use all of GPU memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # create actual tf session
        self._sess = tf.Session(config=config)
        self._sess.as_default()
        self._sess.__enter__()

        loaded = False
        if self.load_model:
            loaded = self.restore(self.conf['save_path_prefix'])

        if not loaded and not self.init_model:
            raise Exception('cannot load existing model')

        if not loaded:
            self._sess.run(tf.global_variables_initializer())

        tf.get_default_graph().finalize()


    def close(self, ty=None, value=None, tb=None):
        '''
        Closes and saves model
        '''

        if self.save_model:
            self.save(self.conf['save_path_prefix'])

        self._sess.__exit__(ty, value, tb)                    
        self._sess.close()
        self._sess = None

