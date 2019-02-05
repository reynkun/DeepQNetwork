import logging

from logging.handlers import TimedRotatingFileHandler

logger = None


def init_logging(save_path_prefix, add_std_err=True):
    global logger
    if logger:
        return logger

    logger = logging.getLogger('rl')
    logger.setLevel(logging.DEBUG)

    log_path = save_path_prefix + '.log'

    hdlr = TimedRotatingFileHandler(log_path, when='D')
    formatter = logging.Formatter('%(asctime).19s [%(levelname)s] %(message)s')
    hdlr.setFormatter(formatter)

    logger.addHandler(hdlr)

    if add_std_err:
        hdlr = logging.StreamHandler()
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

    return logger


def log(*mesg):
    global logger

    if logger:
        logger.info(' '.join([str(m) for m in mesg]))
    else:
        print(' '.join([str(m) for m in mesg]))