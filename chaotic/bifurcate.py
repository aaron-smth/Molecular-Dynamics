import numpy as np
import matplotlib.pyplot as plt

class Chaotic_map:

    def __init__(self, r, x0, truncation=1000, runstep=2000):
        self.r = r
        self.x0 = 0.6
        self.truncation = truncation
        self.runstep = runstep
        self.record = []
        self.run()

    def run(self):
        x = self.x0
        for i in range(self.runstep):
            x = B_map(self.r, x) 
            if i>self.truncation:
                self.record.append(x)

def B_map(r, x):
    return 4 * r * x * (1 - x)

def Lyapunov(r, x0):
    y0 = x0 + 1e-10
    x, y = x0, y0
    lam = 0
    runstep = 2000
    miss_ones = 0
    for i in range(runstep):
        diff1 = np.abs(x-y)
        x,y = B_map(r,x), B_map(r,y) 
        diff2 = np.abs(x-y)
        if diff2 < 1e-40:
            miss_ones += 1
            continue
        lam += np.log( diff2/diff1 )
    lam /= (runstep - miss_ones)
    return lam

def Bifurcate_plot():
    truncation = 1000
    runstep = 2000
    x_N = runstep - truncation - 1
    r_N = 1500
    xx, rr = np.empty( (r_N, x_N) ), np.empty( (r_N, x_N) )
    for i, r in enumerate(np.linspace(0.7,1,r_N)):
        bif = Chaotic_map(r, 0.6, truncation, runstep)
       
        xx[i] = bif.record
        rr[i] = r * np.ones(x_N)

    fig, ax = plt.subplots(figsize=(30,16))
    ax.scatter(rr , xx, c='black', s=1, alpha=0.7)

    fig.savefig('Bifurcate.png')

def Lyapunov_plot():
    x0 = 0.4
    r_arr = np.linspace( 0.7, 1 , 10000)
    lam_li = []
    for r in r_arr:
        lam_li.append( Lyapunov(r, x0) )
    lam_arr = np.array(lam_li)
    fig, ax = plt.subplots(figsize=(30,16))
    ax.plot(r_arr,lam_arr, lw=2)
    ax.scatter(r_arr[lam_arr>0],lam_arr[lam_arr>0], s=10, color='red', 
            label='chaotic')
    inds = np.where(np.diff(lam_arr) > 0.3)[0]
    print(r_arr[inds])
    fig.savefig('Lyapunov.png')

def trajectory(r):
    x0, y0 = 0.6, 0.6 + 1e-10
    x,y  = x0, y0
    x_li, y_li = [], []
    for i in range(2000):
        x_li.append(x)
        y_li.append(y)
        x,y = B_map(r, x), B_map(r, y)
    fig, ax  = plt.subplots()
    xdata = np.arange(2000)
    ax.plot(xdata, x_li, c='b')
    ax.plot(xdata, y_li, c='r')
    plt.show()

if __name__ == '__main__':
    Bifurcate_plot()
    #Lyapunov_plot()
    #trajectory(0.87431621)

