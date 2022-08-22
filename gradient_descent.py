from numpy import *

# y = mx + b
# m(weight) is slope, b(bias) is y-intercept
def error_calculation(b, m, points):
    error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        error += (y - (m * x + b)) ** 2
    return error / float(len(points))

def weight_gradient(b_current, m_current, points, learningRate):
    gradient_b = 0
    gradient_m = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        gradient_b += -(2/N) * (y - ((m_current * x) + b_current))
        gradient_m += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * gradient_b)
    new_m = m_current - (learningRate * gradient_m)
    return [new_b, new_m]

def calculate_gradient(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = weight_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 
    initial_m = 0 
    num_iterations = 1000
    print("Gradient descent of b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, error_calculation(initial_b, initial_m, points)))
    print("starting calculation...") 
    [b, m] = calculate_gradient(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, error_calculation(b, m, points))) 
    

run()