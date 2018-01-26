import yappi
import time

yappi.set_clock_type('cpu')
yappi.start(builtins=True)

start = time.time()



duration = time.time() - start
print('duration is {}'.format(duration))

stats = yappi.get_func_stats()
stats.save('callgrind.out', type='callgrind')
