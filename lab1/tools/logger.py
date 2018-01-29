import logging 
import re

fname = 'results.log'

LOG_FORMAT = '%(asctime)s - %(message)s'
logging.basicConfig(filename=fname, level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()


def logging_number():
    '''Writing head: Experiment []'''
    with open(fname, 'r') as f:
        content = f.readlines()
    if len(content) <= 1:
        logger.info('Experiment 1')
    else:
        pattern = 'Experiment (\d+)'
        for line in reversed(content):
            result = re.search(pattern, line)
            if result:
                n = result.group(1)
                logger.info(f'Experiment {int(n)+1}')


def logging_dict(d):
    import sys
    '''Logging the parameters and outputs of this function'''
    li = [(k,v) for k,v in d.items() if sys.getsizeof(v)<1000 and not callable(v)] # logging varoables that are not too big and not functions
    message = '{}: {}' 
    logger.info(', '.join([message.format(*tup) for tup in li]))
    print('Results Logged')


def logging_time(func):
    '''Recording time'''
    import time 
    def wrapper(*args, **kwargs):
        start = time.time()

        func(*args, **kwargs)        

        duration = time.time() - start
        print(f'duration: {duration}\n')
        logging.info(f'duration : {duration}')
    return wrapper


