import time
import numpy as np

from collections import deque

from .utils.logging import log
from .utils.render import render_game


class GameGenerator:
    DEFAULT_OPTIONS = {
        'frame_skip': 1,
        'use_episodes': True,
        'max_game_length': 50000,
    }


    def __init__(self, conf, env, model):
        self.env = env
        self.model = model

        # set up defaul values
        self.conf = self.DEFAULT_OPTIONS.copy()

        # override saved conf with parameters
        for key, value in conf.items():
            if value is not None and key in self.conf:
                self.conf[key] = value



    def play_generator(self,
                       is_training=False,
                       num_games=None,
                       use_epsilon=False,
                       display=False,
                       save_video=False):
        '''
        Generator which will run the game one frame
        '''

        self.play_init()

        try:
            while num_games is None or self.cur_game_count < num_games:
                for game_state in self.play_game_generator(is_training=is_training, 
                                                           use_epsilon=use_epsilon, 
                                                           display=display, 
                                                           save_video=save_video):
                    yield game_state

        except KeyboardInterrupt:
            log('play interrupted')


    def play_init(self):
        '''
        Initialize play variables
        '''

        # run game settings
        # self.eps_min = self.conf['eps_min']
        # self.eps_max = self.conf['eps_max']
        # self.eps_decay_steps = self.conf['eps_decay_steps']
        # self.discount_rate = self.conf['discount_rate']
        self.use_episodes = self.conf['use_episodes']
        self.frame_skip = self.conf['frame_skip']
        self.max_game_length = self.conf['max_game_length']
        # self.batch_size = self.conf['batch_size']

        # self.total_game_count = self.session.run([self.model.game_count])[0]
        self.total_game_count = self.model.get_game_count()
        self.cur_game_count = 0
        self.step = self.model.get_step()

        # # batch memory
        # self.play_batch = ReplayMemory(self.model.input_height,
        #                                self.model.input_width,
        #                                self.model.input_channels,
        #                                max_size=self.batch_size,
        #                                state_type=self.model.state_type)


        # reporting stats
        self.play_game_scores = deque(maxlen=100)
        self.play_max_qs = deque(maxlen=100)


    def play_game_generator(self,
                            is_training=True,
                            num_games=None,
                            use_epsilon=False,
                            display=False,
                            save_video=False):
        '''
        Generator which yields every frame and runs one game
        '''

        # reset game
        game_state = self.make_game_state()

        while not game_state['game_done']:
            # play out episode
            for _ in self.play_episode_generator(game_state, 
                                                 is_training=is_training,
                                                 use_epsilon=use_epsilon,
                                                 display=display,
                                                 save_video=save_video):
                yield game_state
            
            # update frame final state of episode / game
            game_state['cont'] = 0
            self.update_frame_state(game_state)

            # return to training
            yield game_state

        self.total_game_count += 1
        self.cur_game_count += 1

        self.play_game_scores.append(game_state['score'])
        self.play_max_qs.append(game_state['total_max_q'] / game_state['game_length'])

        self.report_play_stats(game_state, 
                               is_training=is_training)

        if save_video:
            save_path = os.path.join(self.save_dir,
                                     'video-{}-{}.mp4'.format(self.step,
                                                              self.total_game_count))
        else:
            save_path = None

        if save_video or display:
            render_game(game_state['frames_render'],
                        game_state['actions_render'],
                        repeat=False,
                        save_path=save_path,
                        display=display)   

        return game_state


    def play_episode_generator(self, 
                               game_state, 
                               is_training=True, 
                               use_epsilon=False,
                               display=False,                     
                               save_video=False):
        '''
        Generator which yields every frame and runs one episode
        '''

        game_state['episode_length'] = 0
        game_state['episode_done'] = False
        game_state['cont'] = 1

        while not game_state['episode_done'] and not game_state['game_done']:
            self.step = self.model.get_step()

            game_state['game_length'] += 1
            game_state['episode_length'] += 1

            if game_state['game_length'] > self.max_game_length:
                break

            if len(game_state['frames']) >= self.model.num_frames_per_state:
                game_state['cont'] = 1
                self.update_frame_state(game_state)

                # yield to caller
                yield game_state

                # Online DQN evaluates what to do
                q_values = self.model.predict([game_state['state']])

                game_state['total_max_q'] += q_values.max()

                if is_training or use_epsilon:
                    game_state['action'] = self.model.epsilon_greedy(q_values, self.step)
                else:
                    game_state['action'] = np.argmax(q_values)

            game_state['action'] = self.model.before_action(game_state['action'], 
                                                            game_state['observation'], 
                                                            game_state['reward'], 
                                                            game_state['game_done'], 
                                                            game_state['info'])

            # run action for frame_skip steps
            game_state['reward'] = 0
            for i in range(self.frame_skip):
                # online network plays
                game_state['observation'], step_reward, game_state['game_done'], game_state['info'] = self.env.step(game_state['action'])

                if not is_training and (save_video or display):
                    game_state['actions_render'].append(game_state['action'])
                    game_state['frames_render'].append(self.model.render_observation(game_state['observation']))

                game_state['reward'] += step_reward

                # check for episode change
                if self.use_episodes and 'ale.lives' in game_state['info']:
                    num_lives = game_state['info']['ale.lives']

                    if game_state['num_lives'] is not None and num_lives != game_state['num_lives']:
                        game_state['episode_done'] = True

                    if num_lives <= 0:
                        game_state['game_done'] = True

                    game_state['num_lives'] = num_lives

                if game_state['game_done']:
                    break

                if game_state['episode_done']:
                    break

            game_state['score'] += game_state['reward']
            game_state['observation'] = self.model.preprocess_observation(game_state['observation'])
            game_state['frames'].append(game_state['observation'])    


    def report_play_stats(self, game_state, is_training=True):
        '''
        Replay on play stats
        '''

        if not is_training or self.total_game_count % self.model.game_report_interval == 0:
            if game_state['game_length'] > 0:
                mean_max_q = game_state['total_max_q'] / game_state['game_length']
            else:
                mean_max_q = 0

            elapsed = time.time() - game_state['game_start_time']
            if elapsed > 0:
                frame_rate = game_state['game_length'] / (time.time() - game_state['game_start_time'])
            else:
                frame_rate = 0.0

            if len(self.play_game_scores) > 0:
                avg_score = sum(self.play_game_scores) / len(self.play_game_scores)
            else:
                avg_score = 0

            min_score = None
            max_score = None

            for score in self.play_game_scores:
                if max_score is None or score > max_score:
                    max_score = score

                if min_score is None or score < min_score:
                    min_score = score

            if len(self.play_max_qs) > 0:
                avg_max_q = sum(self.play_max_qs) / len(self.play_max_qs)
            else:
                avg_max_q = 0

            epsilon = self.model.epsilon(self.step)

            log('[play] step {} game {}/{} len: {:d} max_q: {:0.3f}/{:0.3f} score: {:0.1f}/{:0.1f}/{:0.1f} eps: {:0.3f} fr: {:0.1f}'.format(
                       self.step,
                       self.cur_game_count,
                       self.total_game_count,
                       game_state['game_length'],
                       mean_max_q,
                       avg_max_q,
                       avg_score,
                       min_score,
                       max_score,
                       epsilon,
                       frame_rate))


    def make_game_state(self):
        '''
        Initialized game state variables
        '''

        game_state = {
            'game_start_time': time.time(),
            'game_done': False,
            'episode_done': False,
            'frames': deque(maxlen=self.model.num_frames_per_state),
            'total_max_q': 0,
            'iteration': 0,
            'game_length': 0,
            'episode_length': 0,
            'score': 0,
            'observation': None,
            'old_state': None,
            'state': None,
            'action': 0,
            'reward': None,
            'cont': 1,
            'info': None,
            'num_lives': None,
            'actions_render': [],
            'frames_render': []
        }        

        game_state['observation'] = self.model.preprocess_observation(self.env.reset())
        game_state['frames'].append(game_state['observation'])

        return game_state


    def update_frame_state(self, game_state):
        '''
        Makes a frame state from 4 sequential frames.
        Also saves memories to replay memory
        '''

        # next_state = self.make_state(game_state['frames'])

        # if is_training and game_state['state'] is not None:
        #     self.add_memories(state=game_state['state'], 
        #                       action=game_state['action'], 
        #                       reward=game_state['reward'], 
        #                       cont=cont, 
        #                       next_state=next_state)

        game_state['old_state'] = game_state['state']
        game_state['state'] = np.concatenate(game_state['frames'], axis=2)


    # def make_state(self, frames):
    #     return np.concatenate(frames, axis=2)
