import logging 

LOG_FORMAT = '%(asctime)s - %(message)s'
logging.basicConfig(filename='results.log', level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()

def logging_dict(d):
    import sys
    '''Logging the parameters and outputs of this function'''
    li = [(k,v) for k,v in d.items() if sys.getsizeof(v)<1000 and not callable(v)] # logging varoables that are not too big and not functions
    message = '{}: {}' 
    logger.info(', '.join([message.format(*tup) for tup in li]))
    print('Results Logged')

def logging_time(func):
    import time 
    def wrapper(*args, **kwargs):
        start = time.time()

        func(*args, **kwargs)        

        duration = time.time() - start
        print('duration: {}\n'.format(duration))
        logging.info('duration : {}'.format(duration)) # logging runtime
    return wrapper


