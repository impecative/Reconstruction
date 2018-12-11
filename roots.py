from __future__ import division, print_function
import numpy as np
from numpy.polynomial import Polynomial as P
import matplotlib.pyplot as plt
import scipy.optimize

def polynomial(t, *constants):
    a, b, c, d, f, g = constants

    t6 = (a*b*c**2*f**4 - a**2*c*d*f**4)  # checked
    t5 = (a**4 + 2*a**2*c**2*g**2 - a**2*d**2*f**4 + b**2*c**2*f**4 + c**4*g**4) # checked
    t4 = (4*a**3*b - 2*a**2*c*d*f**2 + 4*a**2*c*d*g**2 + 2*a*b*c**2*f**2 + 4*a*b*c**2*g**2 - a*b*d**2*f**4 + b**2*c*d*f**4 + 4*c**3*d*g**4) # checked
    t3 = (6*a**2*b**2 - 2*a**2*d**2*f**2 + 2*a**2*d**2*g**2 + 8*a*b*c*d*g**2 + 2*b**2*c**2*f**2 + 2*b**2*c**2*g**2 + 6*c**2*d**2*g**4)  # checked
    t2 = (-a**2*c*d + 4*a*b**3 + a*b*c**2 - 2*a*b*d**2*f**2 + 4*a*b*d**2*g**2 + 2*b**2*c*d*f**2 + 4*b**2*c*d*g**2 + 4*c*d**3*g**4) # checked
    t1 =  (- a**2*d**2 + b**4 + b**2*c**2 + 2*b**2*d**2*g**2 + d**4*g**4) # checked
    t0 =  (- a*b*d**2 + b**2*c*d) # checked

    return np.polyval([t6, t5, t4, t3, t2, t1, t0], t)

def polynomialCoefs(*constants):
    a, b, c, d, f, g = constants

    t6 = (a*b*c**2*f**4 - a**2*c*d*f**4)  # checked
    t5 = (a**4 + 2*a**2*c**2*g**2 - a**2*d**2*f**4 + b**2*c**2*f**4 + c**4*g**4) # checked
    t4 = (4*a**3*b - 2*a**2*c*d*f**2 + 4*a**2*c*d*g**2 + 2*a*b*c**2*f**2 + 4*a*b*c**2*g**2 - a*b*d**2*f**4 + b**2*c*d*f**4 + 4*c**3*d*g**4) # checked
    t3 = (6*a**2*b**2 - 2*a**2*d**2*f**2 + 2*a**2*d**2*g**2 + 8*a*b*c*d*g**2 + 2*b**2*c**2*f**2 + 2*b**2*c**2*g**2 + 6*c**2*d**2*g**4)  # checked
    t2 = (-a**2*c*d + 4*a*b**3 + a*b*c**2 - 2*a*b*d**2*f**2 + 4*a*b*d**2*g**2 + 2*b**2*c*d*f**2 + 4*b**2*c*d*g**2 + 4*c*d**3*g**4) # checked
    t1 =  (- a**2*d**2 + b**4 + b**2*c**2 + 2*b**2*d**2*g**2 + d**4*g**4) # checked
    t0 =  (- a*b*d**2 + b**2*c*d) # checked

    return [t6, t5, t4, t3, t2, t1, t0]


def check(t, a, b, c, d, f, g):
    # a, b, c, d, f, g = constants
    ans = t*((a*t+b)**2 + g**2*(c*t+d)**2)**2 - (a*d-b*c)*(1+f**2*t**2)**2*(a*t+b)*(c*t+d)
    return ans


def main():
    a, b, c, d, f, g = 1, 2, 2, 3, 4, 2
    randint = np.random.randint(-10,10)

    check1 = check(randint, a, b, c, d, f, g)
    check2 = polynomial(randint, a, b, c, d, f, g)

    assert np.isclose(check1, check2), "The expansion doesn't equal the original function for t= {}... something has gone very wrong :(".format(randint)

    # ans1 = check(.1, a, b, c, d, f, g)
    # ans2 = polynomial(.1, a, b, c, d, f, g)
    coefs = polynomialCoefs(a, b, c, d, f, g)
    print("coefficients are t6, t5, t4, ... = ", coefs)

    roots = np.roots(coefs)

    print("The roots of the function are: \n", roots)

    for t in roots:
        print("\nThe expansion yields: {}".format(polynomial(t.real, a, b, c, d, f, g)))
        print("The original function yields: {}\n".format(check(t.real,a, b, c, d, f, g)))

        print("Is the t value a real root? ", np.allclose(polynomial(t.real, a, b, c, d, f, g), 0))

    # print(ans1)
    # print(ans2)

    


# main()

def solvePolynomial(a, b, c, d, f, g):
    # first get the polynomial coefficients so we can find roots
    t6 = (a*b*c**2*f**4 - a**2*c*d*f**4)  # checked
    t5 = (a**4 + 2*a**2*c**2*g**2 - a**2*d**2*f**4 + b**2*c**2*f**4 + c**4*g**4) # checked
    t4 = (4*a**3*b - 2*a**2*c*d*f**2 + 4*a**2*c*d*g**2 + 2*a*b*c**2*f**2 + 4*a*b*c**2*g**2 - a*b*d**2*f**4 + b**2*c*d*f**4 + 4*c**3*d*g**4) # checked
    t3 = (6*a**2*b**2 - 2*a**2*d**2*f**2 + 2*a**2*d**2*g**2 + 8*a*b*c*d*g**2 + 2*b**2*c**2*f**2 + 2*b**2*c**2*g**2 + 6*c**2*d**2*g**4)  # checked
    t2 = (-a**2*c*d + 4*a*b**3 + a*b*c**2 - 2*a*b*d**2*f**2 + 4*a*b*d**2*g**2 + 2*b**2*c*d*f**2 + 4*b**2*c*d*g**2 + 4*c*d**3*g**4) # checked
    t1 =  (- a**2*d**2 + b**4 + b**2*c**2 + 2*b**2*d**2*g**2 + d**4*g**4) # checked
    t0 =  (- a*b*d**2 + b**2*c*d) # checked

    coefs = [t6, t5, t4, t3, t2, t1, t0]

    # find the roots of the sixth order polynomial
    roots = np.roots(coefs)

    # check they are actually roots...

    counter = 0
    for t in roots:
        val = np.polyval(coefs, t.real)
        if np.isclose(val, 0):
            counter += 1

    assert counter > 0, "No roots were found..."


    return roots

roots = solvePolynomial(1,2,2,3,4,2)
