import unittest
from tersoff import *
import numpy as np

def cut_test(r):
    if r>R+D:
        return 0
    elif r>R-D:
        return cut_off(r)
    else:
        return 1

class TestTersoff_V(unittest.TestCase):
    def test_V(self):
        pos1 = np.array(
                [
                [1.,1.,1.],
                [3.,3.,3.]
                ])

        self.assertAlmostEqual(
                 tersoff_V(pos1), 0 )
        self.assertTrue(
                np.allclose( tersoff_F(pos1), np.zeros_like(pos1) ))

        #import ipdb; ipdb.set_trace()   

        pos2= np.array(
                [
                [1.,1.,1.],
                [2.,2.,1.]
                ])
        r2 = pos2[1] - pos2[0]
        d2 = np.linalg.norm(r2)

        self.assertAlmostEqual(
                tersoff_V(pos2), cut_test(d2) * ( fR(d2) - fA(d2))
                ) 

        pos3= np.array(
                [
                [0.,0.,0.],
                [1/2,sqrt(3)/2,0.], 
                [-1/2, sqrt(3)/2,0.]
                ])
        r3 = pos3[1] - pos3[0]
        d3 = np.linalg.norm(r3)

        g3 = 1 + c**2 * (1/ d**2 - 1/ (d**2 + (h-0.5)**2))
        b3 = (1+ (beta * g3)**n)**(-1/(2*n))
        self.assertAlmostEqual(
                tersoff_V(pos3), 3 * cut_test(d3) * ( fR(d3) - b3 * fA(d3))
                ) 
                          
        pos4= np.array(
                [
                [0.,0.,0.],
                [1.,0.,0.], 
                [0.,1.,0.]
                ])
        r4 = pos4[None, :] - pos4[:, None]
        d4 = np.linalg.norm(r4, axis=2)

        b4a = b_eta(g_theta(0))
        b4b = b_eta(g_theta(1/sqrt(2)))

        self.assertAlmostEqual(
                tersoff_V(pos4), 
                ( fR(1) - b4a * fA(1) 
                + fR(sqrt(2)) - b4b * fA(sqrt(2))
                + fR(1) - b4b * fA(1)
                ) )

        pos5= np.array(
                [
                [0.,0.,0.],
                [2.,0.,0.], 
                [0.,1.,0.]
                ])
        b_ = lambda Ctheta: b_eta(g_theta(Ctheta))
        cut5 = cut_test(2)

        self.assertAlmostEqual(
                tersoff_V(pos5), 
                ( fR(1) - b_eta(g_theta(0) * cut5)*fA(1)
                + (fR(2) - b_(0) * fA(2)) * cut5
                + fR(1) - 1 * fA(1)
                + (fR(2) - 1 * fA(2)) * cut5
                ) / 2 )

if __name__ == '__main__':
    unittest.main()
