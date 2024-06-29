# Imports
import scipy.optimize as optimize
import numpy as np
from math import sqrt
import statistics as 

# Helpers
def find_center(x, y):
    """ find the center of a circle given a set of x and y points """
    x_m = np.mean(x)
    y_m = np.mean(y)
    guessed_center = (x_m, y_m)
    center, ier = optimize.leastsq(f_2, guessed_center)
    return center

def calc_R(xc, yc):
    """ calculate the distance of each 2D points from the center c=(xc, yc) """
    return sqrt((x-xc)**2 + (y-yc)**2)

def f_2(c):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(*c)
    return Ri - Ri.mean()

if __name__ == '__main__':
    #   Generate dataset of x and y points
    xpoints = [0.1, 0.2, 0.3]
    ypoints = [0.1, 0.2, 0.3]

    #   use scipy to fit a circle to the data
    center_estimate = (statistics.mean(xpoints), np.mean(ypoints))
    print(center_estimate)

    center_2, ier = optimize.leastsq(f_2, center_estimate)
    print(center_2, ier)

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(*center_2)
    R_2        = Ri_2.mean()
    residu_2   = sum((Ri_2 - R_2)**2)

    print("xc_2, yc_2:", xc_2, yc_2)

    #   calculate new metric based on the circle
