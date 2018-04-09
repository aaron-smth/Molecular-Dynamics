import math

# carbon parameters
m = 12.0107 #u
bondLength = 1.42 #A

# system parameters
#N = 1000
T = 2000
#density = 3.5 # g/cm^3
#L = (m * N / (density / 1.66054))**(1/3) #calculated by density
l = bondLength 
R = 30
L = 2 * R * 3/2
n_defects = 0
dt = 0.01
HeatBath_on = False
memory_on = True
steps = 4000
fromFile = 'sys.obj'
  




def is_info(k, v):
    if k.startswith('_'):
        return False
    elif type(v) in [int, float, bool, str]:
        return True
    elif v == None:
        return True
    else:
        return False
custom_info = {k:v for k,v in globals().items() if is_info(k, v)}
        
