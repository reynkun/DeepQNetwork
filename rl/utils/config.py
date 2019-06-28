import os
import json


def get_conf(save_dir):
    '''
    Will find and read json-formatted conf from save dir
    '''
    conf = None

    # find conf file in save_dir
    conf_count = 0
    for fn in os.listdir(save_dir):
        if fn.endswith('.conf'):
            with open(os.path.join(save_dir, fn)) as fin:
                conf = json.load(fin)
                conf_count += 1

    if conf_count > 1:
        raise Exception('too many confs in directory')

    return conf