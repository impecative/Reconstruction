from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from algorithm12_1 import *

def polynomial(t, *constants):
    a, b, c, d, f, g = constants

    t6 = np.longdouble(a*b*c**2*f**4 - a**2*c*d*f**4)
    t5 = np.longdouble(a**4 + 2*a**2*c**2*g**2 - a**2*d**2*f**4 + b**2*c**2*f**4 + c**4*g**4)
    t4 = np.longdouble(4*a**3*b - 2*a**2*c*d*f**2 + 4*a**2*c*d*g**2* + 2*a*b*c**2*f**2 + 4*a*b*c**2*g**2 - a*b*d**2*f**4 + b**2*c*d*f**4 + 4*c**3*d*g**4)
    t3 = np.longdouble(6*a**2*b**2 - 2*a**2*d**2*f**2 + 2*a**2*d**2*g**2 + 8*a*b*c*d*g**2 + 2*b**2*c**2*f**2 + 2*b**2*c**2*g**2 + 6*c**2*d**2*g**4)
    t2 = np.longdouble(- a**2*c*d +  4*a*b**3 +  a*b*c**2 - 2*a*b*d**2*f**2 + 4*a*b*d**2*g**2 + 2*b**2*c*d*f**2 + 4*b**2*c*d*g**2 + 4*c*d**3*g**4 + a*b*c**2)
    t1 =  np.longdouble(- a**2*d**2 + b**4 + b**2*c**2 + 2*b**2*d**2*g**2 + d**4*g**4)
    t0 =  np.longdouble(- a*b*d**2 + b**2*c*d)

    one = np.longdouble(a**4*t**5 + 4*a**3*b*t**4 + 6*a**2*b**2*t**3 + 2*a**2*c**2*g**2*t**5 -a**2*c*d*f**4*t**6 - 2*a**2*c*d*f**2*t**4)
    two = np.longdouble(4*a**2*c*d*g**2*t**4 - a**2*c*d*t**2 - a**2*d**2*f**4*t**5 - 2*a**2*d**2*f**2*t**3 + 2*a**2*d**2*g**2*t**3 - a**2*d**2*t)
    three = np.longdouble(4*a*b**3*t**2 + a*b*c**2*f**4*t**6 + 2*a*b*c**2*f**2*t**4 + 4*a*b*c**2*g**2*t**4 + a*b*c**2*t**2 + 8*a*b*c*d*g**2*t**3)
    four = np.longdouble(- a*b*d**2*f**4*t**4 - 2*a*b*d**2*f**2*t**2 + 4*a*b*d**2*g**2*t**2 - a*b*d**2 - a**2*d**2*t + b**2*c**2*f**4*t**5)
    five = np.longdouble(2*b**2*c**2*f**2*t**3 + 2*b**2*c**2*g**2*t**3 + b**2*c**2*t + b**2*c*d*f**4*t**4 + 2*b**2*c*d*f**2*t**2 + 4*b**2*c*d*g**2*t**2)
    six = np.longdouble(b**2*c*d + 2*b**2*d**2*g**2*t + c**4*g**4*t**5 + 4*c**3*d*g**4*t**4 + 6*c**2*d**2*g**4*t**3 + 4*c*d**3*g**4*t**2 + d**4*g**4*t)

    total = one+two+three+four+five+six

    return total#(t6 * t**6) + (t5 * t**5) + (t4 * t**4) + (t3 * t**3) + (t2 * t**2) + t1*t + t0, total

def check(t, *constants):
    a, b, c, d, f, g = constants

    # term1 = t * ((a*t+b)**2 + g**2*(c*t+d)**2)**2
    # term2 = (a*d-b*c)*(1+f**2*t**2)**2 * (a*t+b)*(c*t+d)

    return t*((a*t+b)**2 + g**2*(c*t+d)**2)**2 - (a*d-b*c)*(1+f**2*t**2)**2*(a*t+b)*(c*t+d)#, term1-term2

def polynomialCoefs(*constants):
    a, b, c, d, f, g = constants

    t6 = a*b*c**2*f**4 - a**2*c*d*f**4
    t5 = a**4 + 2*a**2*c**2*g**2 - a**2*d**2*f**4 + b**2*c**2*f**4 + c**4*g**4
    t4 = 4*a**3*b - 2*a**2*c*d*f**2 + 4*a**2*c*d*g**2* + 2*a*b*c**2*f**2 + 4*a*b*c**2*g**2 - a*b*d**2*f**4 + b**2*c*d*f**4 + 4*c**3*d*g**4
    t3 = 6*a**2*b**2 - 2*a**2*d**2*f**2 + 2*a**2*d**2*g**2 + 8*a*b*c*d*g**2 + 2*b**2*c**2*f**2 + 2*b**2*c**2*g**2 + 6*c**2*d**2*g**4
    t2 = - a**2*c*d +  4*a*b**3 +  a*b*c**2 - 2*a*b*d**2*f**2 + 4*a*b*d**2*g**2 + 2*b**2*c*d*f**2 + 4*b**2*c*d*g**2 + 4*c*d**3*g**4
    t1 =  - a**2*d**2 - a**2*d**2 + b**2*c**2 + d**4*g**4 + 2*b**2*d**2*g**2
    t0 =  - a*b*d**2 + b**2*c*d

    return [t6, t5, t4, t3, t2, t1, t0]


# constants = (1,1,1,1,1,1)
# p = polynomialCoefs(1,1,1,1,1,1)
# roots = np.roots(p)
# print("Roots of the sixth order polynomial: ", roots)

# print("Is this close to zero?", check(roots[0], 1,1,1,1,1,1))

# sol = optimize.fsolve(check, roots[3].real, args=constants)
# print(sol)
# print(polynomial(sol, 1,1,1,1,1,1))
# print(check(sol, 1,1,1,1,1,1))

# # print(Term1(roots2[1], 2,1,4,2,1))

constants = (2,1,3,4,3,7)
# print(check(100, 2, 1, 3, 4,3 ,7))

p = polynomialCoefs(2,1,3,4,3,7)
roots = np.roots(p)

newroots = []

# sol = optimize.fsolve(check, )

for t in roots:
    newroots.append(optimize.fsolve(check, t.real, args=constants,))

# for t in newroots:
#     print(check(t, 2,1,3,4,3,7))

print(newroots)

# plt.figure()

# xmin, xmax = np.min(newroots), np.max(newroots)

# t = np.linspace(xmin, xmax, np.round((xmax-xmin)*10000))
# y = check(t, 2,1,3,4,3,7)

# plt.plot(t,y, "b-")

# for root in newroots:
#     plt.plot(root, check(root, 2,1,3,4,3,7), "ro")

# plt.xlabel("t")
# plt.ylabel("g(t)")

# plt.show()
















def costFunction(t, a, b, c, d, f, g):

    return (t**2/(1+f**2*t**2)) + (c*t+d)**2/((a*t+b)**2 + g**2*(c*t+d)**2) 

def evaluateCostFunction(roots, a, b, c, d, f, g):
    """Evaluate the cost function (12.5) for input array/list np.array([t1,t2, ...]).\n
    Select the value of t for which the cost function is the smallest."""

    tmin = 999999
    minCostFn = 99999999
    
    for t in roots:
        costFn = costFunction(t, a, b, c, d, f, g)
        if costFn < minCostFn:
            minCostFn = costFn
            tmin = t
        else:
            pass
    
    assert tmin != 999999, "No minimum value of t has been found..."
    assert minCostFn != 99999999, "The cost function has not been minimised..."

    # also find the value of cost function as t=infty, corresponding to 
    # an epipolar line fx=1 in the first image .

    inftyCostFn = (1/f**2) + (c**2/(a**2 + g**2*c**2))

    assert inftyCostFn > minCostFn, "The epipolar line in the first image is fx=1, tmin=infty"

    return tmin

# tmin = evaluateCostFunction(newroots, 2,1,3,4,3,7)
# print(newroots)
# print(tmin)

def findModelPoints(tmin, a, b, c, d, f, g ):
    """Find the model points x and x' that fit the epipolar constrant x'^T F x = 0"""

    l1 = np.array([tmin*f, 1, -tmin])
    l2 = np.array([-g*(c*t+d), a*t+b, c*t+d])

    x1 = np.array([-l1[0]*l1[2], -l1[1]*l1[2], l1[0]**2 + l1[1]**2])
    x2 = np.array([-l2[0]*l2[2], -l2[1]*l2[2], l2[0]**2 + l2[1]**2])

    return x1, x2

# x1, x2 = findModelPoints(tmin, 2,1,3,4,3,7)
# print(x1, x2)
# x1 = homogeneous2Inhomogenous(x1)
# x2 = homogeneous2Inhomogenous(x2)

# print("Coordinate 1 is \n", x1)
# print("\nCoordinate 2 is \n", x2)


