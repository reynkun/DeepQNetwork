import logging

from logging.handlers import TimedRotatingFileHandler


logger = None


def init_logging(save_path=None, add_std_err=True):
    '''
    Initialize singleton style logger
    '''

    global logger
    if logger:
        return logger

    logger = logging.getLogger('rl')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime).19s [%(levelname)s] %(message)s')

    if save_path is not None:
        log_path = save_path + '.log'
        hdlr = TimedRotatingFileHandler(log_path, when='D')
        hdlr.setFormatter(formatter)

        logger.addHandler(hdlr)

    if add_std_err:
        hdlr = logging.StreamHandler()
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

    return logger


def log(*mesg):
    '''
    Log a message
    '''

    global logger

    if logger:
        logger.info(' '.join([str(m) for m in mesg]))
    else:
        print(' '.join([str(m) for m in mesg]))