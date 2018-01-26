import yappi
import time
from lab1 import  plot_linear

yappi.set_clock_type('cpu')
yappi.start(builtins=True)

start = time.time()

plot_linear()

duration = time.time() - start
print('duration is {}'.format(duration))

stats = yappi.get_func_stats()
stats.save('callgrind.out', type='callgrind')
