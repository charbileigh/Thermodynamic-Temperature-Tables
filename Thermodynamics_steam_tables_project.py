"""
Name : Francesca Seopa
Student Number: SPXMMA001
Project 2: MEC3075F - Computer Methods for Mechanical Engineering
Date Due : 17th June 2020, 17h00

"""


# importing built in functions to perform calculations  for this project
# Some functions are used to plot 2-D graphs and others 3-D graphs
import scipy as sci
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from math import sqrt
from pprint import pprint
from csv import reader
import scipy
import math
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D




# Reading the npz files for calculations
# As a means of calculating the K before getting
# the assemble_k() function
data = np.load('K3.npz')
lst = data.files
K = None
for item in lst:
    print(item)
    K = data[item]
A = K

  
# The beginning of the calculations for the project
# Steps for function get_cases
def get_cases():
    '''

    This function is used to extract case numbers that correspond
    to a student number and it's peoplesoft number.
    There are 4 case numbers which will be used as indicators for
    getting the temperature parameters used for the project.

    '''
    with open('allocation.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        i = 0
        cases = []
        # Pass reader object to list() to get a list of lists:
        for row in csv_reader:   
            i = i + 1
            if i == 55:
                return row
            
print('Cases:')
print(get_cases())




#steps for function get_parameters
def get_parameter():
    '''

    Parameters are obtained through the case numbers corresponding
    to the peoplesoft number. The parameters are the temperature
    boundaries used for calculations and plots for sets 1 and 2

    '''
    with open('parameters.csv','r') as parameter:
        j = 0
        parameters = [] 
        csv_get = reader(parameter)
        for row in csv_get:
            j = j + 1
            if j == 7:
               set1 = parameters.append(row[1:])
            if j == 18:
               set2= parameters.append(row[1:])
            if j == 37:
                set3 = parameters.append(row[1:])
            if j == 39:
                set4 = parameters.append(row[1:])
        return parameters

print('Parameters:')
print(get_parameter())




def assembly_K(n, m):
    '''

    This function is used to calculate the K matrix of any size.
    The K matrix will be used as the A in the equation Ax = b.
    To solve for the temperature graphs for set 1 and set2

    '''
    K = np.zeros((n,m))
    square = int(math.sqrt(n))-1
    counter_1 = 0
    counter_2 = 0

    for i in range(0, n):
        for j in range(0, m):
            if j == i:
                K[i, j] = 4

            elif j == (i+1):
                if counter_1 == square:
                    K[i, j] = 0
                    counter_1 = 0
                else:
                    K[i, j] = -1
                    counter_1 += 1

            elif j == (i-1):
                if counter_2 == square:
                    K[i, j] = 0
                    counter_2 = 0
                else:   
                    K[i, j] = -1
                    counter_2 += 1

            elif j == (i+3):
                K[i, j] = -1

            elif j == (i-3):
                K[i, j] = -1
    return K






def cholesky(A):
    """

    Performs a Cholesky decomposition of A, which must 
    be a symmetric and positive definite matrix. The function
    returns the lower variant triangular matrix, L.

    """
    n = len(A)

    # Create zero matrix for L
    L = [[0.0] * n for i in range(n)]

    # Perform the Cholesky decomposition
    for i in range(n):
        for k in range(i+1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
            
            if (i == k): # Diagonal elements
                # LaTeX: l_{kk} = \sqrt{ a_{kk} - \sum^{k-1}_{j=1} l^2_{kj}}
                L[i][k] = sqrt(A[i][i] - tmp_sum)
            else:
                # LaTeX: l_{ik} = \frac{1}{l_{kk}} \left( a_{ik} - \sum^{k-1}_{j=1} l_{ij} l_{kj} \right)
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))
    return L




 
def solveLU(L,U,b):
    '''

    The second part of the  def cholesky(A) Function
    Where the L obtained above is used to solve for the solution
    using the forward and backward substitution methods

    '''
    L = np.array(L,float)
    U = np.array(U,float)
    b = np.array(b,float)
    n,_= np.shape(L)
    y = np.zeros(n)
    x = np.zeros(n)

    # Forward Substitution
    for i in range(n):
        sumj = 0
        for j in range(i):
            sumj += L[i,j] * y[j]
        y[i] = (b[i] - sumj)/L[i,i]

    # backward substitution
    for i in range(n-1,-1,-1):
        sumj = 0
        for j in range(i+1,n):
            sumj += U[i,j] * x[j]
        x[i] = (y[i]-sumj)/U[i,i]
    return x


def jacobi(A, b, N=10, tol=1e-9):
    '''

    is an iterative algorithm for determining the solutions of
    a strictly diagonally dominant system of linear equations.
    Each diagonal element is solved for, and an approximate value is plugged in.
    The process is then iterated until it converges.
    Used for calculating the entries for set 2.

    '''
    x_0 = np.zeros(len(b))
    diagonal_matrix = np.diag(np.diag(A))
    lower_upper_matrix = A - diagonal_matrix
    k = 0
    while k < N:
        k = k + 1
        diagonal_matrix_inv = np.diag(1 / np.diag(diagonal_matrix))
        x_new = np.dot(diagonal_matrix_inv, b - np.dot(lower_upper_matrix, x_0))
        if np.linalg.norm(x_new - x_0) < tol:
            return x_new
        x_0 = x_new

    return x_0



# print(np.linalg.solve(A,b))
## Calling the cases obtained in the function

k_set1 = assembly_K(4,4)        #calling the K assembly matrix
L = cholesky(k_set1)
print("L:")
print(L)


b = [['-97', '11', '-41', '10'], ['-48', '4', '15', '7'], ['-47', '5', '63', '3']]


sol = []
for i in b:
    # Loops around the different b values for set1.1, set1.2 and set1.3
    # The sets will be used for plotting set 1.
    x = solveLU(L, np.transpose(L), i)
    sol.append(x)
    print(x)




# plotting for solution of Set 1
# The plot array numbers were obtained through the cholesky function,
# and the forward and backward substitution methods
plot = np.array([[-27.07655502,4.01913876,-12.07177033,-7.28708134],[-12.66985646,-2.16746411,3.62200957,-0.51196172],[-11.69856459,-1.67464115,16.22009569,1.88038278]])
sns.heatmap(plot, linewidth = 0.3, annot = True, cmap = "PRGn",cbar_kws = {'label':'Temperature'})
plt.title("The Temperature Heat Map")
plt.xlabel('Length of Plate')
plt.ylabel('Width of Plate')
plt.show()


# Calling all the individual cases that were given in the cases CV file
# These values are called individually, instead of an array
cases = get_cases()
ta = cases[1]
tb = cases[2]
tc = cases[3]
td = cases[4]


A = assembly_K(4,4)         # Calling the K assembly matrix
print("A:")
print(A)

# The last array from the parameter() fucntion is used to calculate the Jacobi
b1 = np.array([97, 4, 65, 8])
print(b1)
b2 = jacobi(A, b1)          # Calling the Jacobi() Function
print(b2)
b = b2

# The values of the new b were obtained by using the Jacobi function to get b2
# The values of B2 where then written below manually
b = [[29.98684311,8.49657345,19.86381149,14.46244144]]



# These are the constraints used to plot the sin and cos graph of the temperature
# Coefficients to get a 3 Dimensional representation of the data
# These instructions are used to plot Set 2
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-1, 1, 0.001)
Y = np.arange(-1, 1, 0.001)
X, Y = np.meshgrid(X, Y)
Z = np.sin(4*X)*97 + np.cos(65*Y)*8
surface = ax.plot_surface(X, Y, Z, linewidth = 0, antialiased = True, cmap = cm.PRGn)


# Customize the z axis.
ax.set_zlim(-100, 100)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


# Add a color bar which maps values to colors.
fig.colorbar(surface, shrink = 0.5, aspect = 5)
plt.show()
