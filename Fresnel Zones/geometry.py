import numpy as np
import matplotlib.pyplot as plt

def path_difference(A, B, X, Y):
    Ax, Ay = A; Bx, By = B
    L = np.linalg.norm(B - A)
    path_diff = np.sqrt((Ax - X)**2 + (Ay - Y)**2) + np.sqrt((Bx - X)**2 + (By - Y)**2) - L
    return path_diff