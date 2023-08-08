from sympy import *
import random

"""Date: 08/08/2023
   Author: Ashe Vazquez
   This program is executes a generic descent rate algorithm to find the minima of the given function, input as a 
   sympy expression."""

def start_point(degree, lower_bound, upper_bound):
    """this function randomly generates the starting point of the GD algorithm"""
    random_list = []
    for i in range(degree + 1):
        n = random.randint(lower_bound, upper_bound)  # starting coefficients between -10 and 10
        random_list.append(n)
    return random_list


# forgot numpy exists
def mult_vector(scalar, lst):
    ret = []
    for index in range(len(lst)):
        ret.append(lst[index]*scalar)
    return ret


def sum_vector(lst1, lst2):  # how the fuck doesn't sympy have an easy method for this
    """sums lists representing vectors in R^n"""
    ret = []
    for index in range(len(lst1)):
        ret.append(lst1[index]+lst2[index])
    return ret


def list_for_sub(vars, point):
    list_to_sub = []
    for index in range(len(vars)):
        list_to_sub.append((vars[index], point[index]))
    return list_to_sub


def calculate_gradient(function, vars, point):
    """returns the gradient (vector size len(vars)) of the given expression at the given point. point is
    a list. of length len(vars) representing a point in len(vars) dimensional space."""
    grad = []
    for var in range(len(vars)):
        partial_derivative = diff(function, vars[var])
        list_to_sub = list_for_sub(vars, point)
        grad.append(partial_derivative.subs(list_to_sub))
    return grad


def grad_descent_algorithm(function, vars, point, learning_rate, desired_runs, runs_so_far):
    runs_so_far += 1
    print(point)
    print(runs_so_far)
    if runs_so_far > desired_runs:
        return point
    else:
        gradient_at_point = calculate_gradient(function, vars, point)
        # point is subtracted by gradient_at_point
        next_point = sum_vector(point, mult_vector(-1*learning_rate, gradient_at_point))
        return grad_descent_algorithm(function, vars, next_point, learning_rate, desired_runs, runs_so_far)


# example run
x = Symbol('x')
fn = sympify('x**2 - 2*x')

lower_bound = -5
upper_bound = 5

point = [random.randint(lower_bound, upper_bound)]
vars = {0: 'x'}
lr = 0.2
desired_r = 30

grad_descent_algorithm(fn, vars, point, lr, desired_r, 0)

plot( fn, (x, -3, 3))