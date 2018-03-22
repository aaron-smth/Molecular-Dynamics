

#tersoff parameters
A = 1393.6 #ev
B = 430. #old:346.74 
lam1 = 3.4879 #A^-1
lam2 = 2.2119
lam3 = 0 
n = 0.72751 #1
c = 38049.0
d = 4.3484
h = -0.930 #old:-0.57058
beta= 1.5724e-7
R = 1.95 #A
D = 0.15

# carbon parameters
m = 12.0107 #u
bondLength = 1.42 #A

# system parameters
N = 100
T = 300
L = bondLength * N**(1/3) * 1. #A, boxsize
dt = 0.01
HeatBath_on = True
memory_on = False
steps = 2000
fromFile = None #'sys.obj' 
