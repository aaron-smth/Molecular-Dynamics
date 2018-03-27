# carbon parameters
m = 12.0107 #u
bondLength = 1.42 #A

# system parameters
N = 1000
T = 300
density = 3.5 # g/cm^3
L = (m * N / (density / 1.66054))**(1/3) #calculated by density
dt = 0.01
HeatBath_on = True
memory_on = False
steps = 2000
fromFile = 'sys.obj' 
    
custom_info = globals()
        
