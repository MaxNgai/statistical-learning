import matplotlib.pyplot as plt
import numpy as np



def style():
    plt.style.use('_mpl-gallery') #grid style

# dot to line
def dotToCurve(x, y):
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    ax.legend()
    plt.show()

# scatter
def scatter(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x,y)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    ax.legend()
    plt.show()

def scatter3D(x,y,z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(x,y,z)
    ax.legend()
    plt.show()

def bar(x,y):
    fig, ax = plt.subplots()
    ax.bar(x,y)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    ax.legend()
    plt.show()

def hist(x):
    fig, ax = plt.subplots()
    ax.hist(x, edgecolor="white")
    plt.xlabel('X')
    plt.ylabel('frequency')
    ax.legend()
    plt.show()

def boxplot(x):
    # x is an array of array
    fig, ax = plt.subplots()
    ax.boxplot(x)
  
    ax.legend()
    plt.show()


def residualScatter(y, ybar):
    residual = np.abs(y-ybar)

    fig, ax = plt.subplots()
    ax.scatter(ybar,residual)
    plt.xlabel('yBar')
    plt.ylabel('Residual')
    plt.show()


