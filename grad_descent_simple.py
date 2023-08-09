"""Date: 08/08/2023
   Author: Ashe Vazquez
   This program is executes a generic descent rate algorithm to find the minima of the given function, input as a 
   sympy expression."""

from sympy import *
import random
import numpy as np


def start_point(degree, lower_bound, upper_bound):
    """this function randomly generates the starting point of the GD algorithm"""
    random_list = []
    for i in range(degree + 1):
        n = random.randint(lower_bound, upper_bound)  #
        random_list.append(n)
    return random_list


def list_for_sub(vars, point):
    """This function receives a dictionary in the form {0: a_0, 1: a_1, ..., n: a_n} and a point which is a list
     representing a point in R^n, and returns a list of tuples which associates each element of the point to
     the corresponding variable name. This list will then be input into the sympy subs method."""
    list_to_sub = []
    for index in range(len(vars)):
        list_to_sub.append((vars[index], point[index]))
    return list_to_sub


def calculate_gradient(function, vars, point):
    """returns the gradient (vector size len(vars)) of the given expression at the given point. point is
    a list of length len(vars) representing a point in len(vars) dimensional space."""
    grad = []
    for var in range(len(vars)):
        partial_derivative = diff(function, vars[var])
        list_to_sub = list_for_sub(vars, point)
        grad.append(partial_derivative.subs(list_to_sub))
    return grad


def grad_descent_algorithm(function, vars, point, learning_rate, desired_runs, runs_so_far):
    runs_so_far += 1
    if runs_so_far > desired_runs:
        return point
    else:
        gradient_at_point = calculate_gradient(function, vars, point)
        # point is subtracted by gradient_at_point
        next_point = np.array(point) - np.array(gradient_at_point) * learning_rate
        return grad_descent_algorithm(function, vars, next_point, learning_rate, desired_runs, runs_so_far)


# example run
x = Symbol('x')  # make all the sympy symbols you're going to need to run the algo
fn = sympify('x**2 - 2*x')  # this is the expression we're going to find the minimum of. Make this differentiable!

# these bounds determine the range from which the starting_point for the algorithm will be selected from.
lower_bound = -5
upper_bound = 5

starting_point = [random.randint(lower_bound, upper_bound)]
vars = {0: 'x'}
lr = 0.2
desired_runs = 30

grad_descent_algorithm(fn, vars, starting_point, lr, desired_runs, 0)

plot( fn, (x, -3, 3))