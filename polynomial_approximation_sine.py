"""Date: 04/08/2023
   Author: Ashe Vazquez
This program uses gradient descent to find an n-degree polynomial which approximates the sine function between -3 and
3. This exercise problem is found on
https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/optimizing-multivariable-functions/a/what-is-gradient-descent

At this current point I haven't yet verified whether the polynomial this code spits out actually approximates a sine
function - I think it doesn't  :(("""

from sympy import *
import random
import numpy as np


def start_point(degree, lower_bound, upper_bound):
    """this function randomly generates the starting point of the GD algorithm"""
    random_list = []
    for i in range(degree + 1):
        n = random.randint(lower_bound, upper_bound)  # starting coefficients between -10 and 10
        random_list.append(n)
    return random_list


def list_for_sub(vars, point):
    list_to_sub = []
    for index in range(len(vars)):
        list_to_sub.append((vars[index], point[index]))
    return list_to_sub


def calculate_gradient(function, vars, point):
    """returns the gradient (vector size len(vars)) of the given expression at the given point. point is
    a list. of length len(vars) representing a point in len(vars) dimensional space."""
    grad = []
    list_to_sub = list_for_sub(vars, point)
    print(list_to_sub)
    for var in range(len(vars)):
        print(vars[var])
        partial_derivative = diff(function, vars[var])
        evaluated_pd = partial_derivative.subs(list_to_sub)
        grad.append(evaluated_pd)
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


deg = 5  # degree of our polynomial

# bounds for the starting point for our polynomial
lower_bound = -10
upper_bound = 10

lr = 0.2  # learning rate

desired_runs = 100  # how many times we want to iterate the algorithm

# vars is a dictionary containing variable names a_0, a_1, ...., a_5 as keys for values 0, 1, ..., 5
vars = {}

# generation expression caled summary_function. this is what we'll minimise
string_polynomial = ""  # string object
x = Symbol('x')
for i in range(deg+1):
    vars[i] = Symbol('a_'+str(i))
    string_polynomial += "+" + str(vars[i])+"*x**"+str(i)
expr = sympify(string_polynomial)  # finished polynomial expression


# this expression, when evaluated for specific values for a_0, ..., a_5, will return a real number which is what this
# program is trying to minimize.
summary_function = integrate((expr - sin(x))**2, (x, -3, 3))
print(summary_function)

# generation of starting point
point = start_point(deg, lower_bound, upper_bound)
print('this is the starting point', point)

min = grad_descent_algorithm(summary_function, vars, point, lr, desired_runs, 0)  # determining min
print(min)

# returning min as floating pt numbers
flt = []
for index in range(len(min)):
    flt.append(N(min[index]))
print('this is the minimum', flt)

optimal_polynomial = summary_function.subs(list_for_sub(vars, flt))

plot(optimal_polynomial, (x, -3, 3))
