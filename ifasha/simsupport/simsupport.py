#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import numpy as np
import math

#This code can factor any number of any length... eventually.
#This is because python has built in arbitrarily length numbers.
def prime_factors(x):
    """Function used to compute all prime factors of a given number, which is an int bigger than 1

    Args:
        x (int) : the given number

    Returns:
        list: all the prime factors of x

    """
    factors = []
    i = 2
    while x > 1:
        if x % i == 0:
            x = x / i
            factors.append(i)
        else:
            i += 1
    return factors

# computes all factors of a given number
def all_factors(n):
    """Function used to compute all factors of a given number, which is an int bigger than 1

    Args:
        x (int) : the given number

    Returns:
        set: all the factors of x

    """
    return set(reduce(list.__add__,([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def best_partition(N,X,Y):
    """Function used to determine the best partition of a give number, N,
    with the intended row number, X, and the column number, Y.

    Args:
        N (int): the number of total partitions
        X (int): intended row number
        Y (int): intended column number

    Returns:
        list: best partition of N. The entries are the best row and column numbers, based on the intended X and Y

    Examples:
        >>> print (best_partition(30,7,4))
        [6, 5.0]

        >>> print (best_partition(20,7,4))
        [5, 4.0]

    """
    Nx_ideal = math.sqrt(N*X/Y)
    # Modified by zezhou here, no function named computeAllFactors()
    # factors = computeAllFactors(N)
    factors = all_factors(N)

    Nx = min(factors, key=lambda x:abs(x-Nx_ideal))
    
    return [Nx, N/Nx]