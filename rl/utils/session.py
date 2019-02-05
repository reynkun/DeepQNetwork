import os
import importlib
import gym
import time
import tensorflow as tf


from .logging import log


class Session:
    def __init__(self, 
                 conf, 
                 init_env=True, 
                 init_model=False, 
                 load_model=True,
                 save_model=False):
        self.conf = conf
        self.init_env = init_env
        self.init_model = init_model
        self.load_model = load_model
        self.save_model = save_model


    def __enter__(self):
        return self.open()


    def __exit__(self, ty, value, tb):
        self.close(ty, value, tb)


    def run(self, *args, **kwargs):
        return self._sess.run(*args, **kwargs)


    def save(self, save_path_prefix):
        log('saving model: ', save_path_prefix)
        self.saver.save(self._sess, save_path_prefix)
        log('saved model')


    def restore(self, save_path_prefix):
        if not os.path.exists(save_path_prefix + '.index'):
            log('  model does not exist:', save_path_prefix)
            return False

        log('  restoring model: ', save_path_prefix)
        self.saver.restore(self._sess, save_path_prefix)
        log('  restored model')

        return True


    def open(self):
        log('creating new session. init_env:', self.init_env, 'load_model:', self.load_model, 'save_model:', self.save_model)

        if self.init_env:
            self.env = gym.make(self.conf['game_id'])
            self.env.seed(int(time.time()))
            self.conf['action_space'] = self.env.action_space.n

        tf.reset_default_graph()

        # make agent
        if isinstance(self.conf['agent'], str):
            mod_agent_str, cl_agent_str = self.conf['agent'].rsplit('.', 1)
            mod_ag = importlib.import_module(mod_agent_str)

            agent_class = getattr(mod_ag, cl_agent_str)
        elif inspect.isclass(self.conf['agent']):
            agent_class = self.conf['agent']
        else:
            raise Exception('invalid agent class')

        self.model = agent_class(conf=self.conf)

        # make saver
        self.saver = tf.train.Saver()

        # configure tensorflow
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

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

        return self


    def close(self, ty=None, value=None, tb=None):
        if self.init_env:
            self.env.close()

        if self.save_model:
            self.save(self.conf['save_path_prefix'])

        self._sess.__exit__(ty, value, tb)                    
        self._sess.close()
        self._sess = None

