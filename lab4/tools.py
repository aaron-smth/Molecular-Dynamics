import logging 
import numpy as np
import re
import os

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
                logger.info('Experiment {}'.format(int(n)+1))


def logging_dict(d):
    import sys
    '''Logging the parameters and outputs of this function
    the sys.getsizeof function is to make sure large arrays don't get logged '''
    li = [(k,v) for k,v in d.items() if type(v) in (int, float)] 
    message = '{}: {}' 
    logger.info('\n' + '\n'.join([message.format(*tup) for tup in li]) +'\n')
    for k,v in li:
        print(f'{k}: {v}')
    print('Results Logged\n')


def logging_time(func):
    '''Recording time'''
    import time 
    def wrapper(*args, **kwargs):
        start = time.time()

        func(*args, **kwargs)        

        duration = time.time() - start
        print('duration: {}\n'.format(duration))
        logging.info('duration : {}'.format(duration)) # logging runtime
    return wrapper

def progress_bar(count, total, status=''):
    import sys
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def to_xyz(f, description, arr, name='Ar', dfmt='%.5f'): 

    N, dim = arr.shape 

    '''writing head'''
    head = np.array([[str(N)],[description]])

    '''remember to use write binary mode '''
    np.savetxt(f, head, fmt='%s')
    np.savetxt(f, arr,  fmt=f'{name}' + f' {dfmt}'*dim)

