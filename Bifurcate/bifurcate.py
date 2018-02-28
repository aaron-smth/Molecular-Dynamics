import numpy as np
import matplotlib.pyplot as plt

class Bifurcate:

    truncation = 1000
    runstep = 3000
    def __init__(self, r, x0):
        self.r = r
        self.x0 = x0
        self.record = []
        self.run()

    def B_map(self, x):
        return 4 * self.r * x * (1 - x)

    def run(self):
        x = self.x0
        for i in range(self.runstep):
            x = self.B_map(x) 
            if i>self.truncation:
                self.record.append(x)

x_N = 1999 
r_N = 1500
xx, rr = np.empty( (r_N, x_N) ), np.empty( (r_N, x_N) )
for i, r in enumerate(np.linspace(0.7,1,r_N)):
    bif = Bifurcate(r, 0.6)
   
    xx[i] = bif.record
    rr[i] = r * np.ones(x_N)

fig, ax = plt.subplots(figsize=(30,16))
ax.scatter(rr , xx, s=1, c='black', alpha=0.7)

fig.savefig('colormap.png')
