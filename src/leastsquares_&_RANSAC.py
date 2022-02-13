#!/usr/bin/python3

from inspect import Parameter
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ball_trajectory_fitting import standard_least_squares,get_best_fit_line
from math import log
import random

def fit_linear_least_squares(age,charges):
    ones=np.ones(len(age))
    # print(ones.shape)
    age_augmented=np.vstack((age,ones)).T
    # print(age_augmented.shape)

    res=standard_least_squares(age_augmented,charges)
    print("LLS params:",res)
    # points=res[0]*age_augmented[:,0]+res[0]*age_augmented[:,1]
    points=get_best_fit_line(age,res,degree=1)

    # return age,points
    return points

# def fit_total_least_squares(x_array,y_array):

#     x_mean=np.mean(x_array)
#     y_mean=np.mean(y_array)
#     U=np.vstack((x_array-x_mean,y_array-y_mean)).T
#     C = np.dot(U.T,U)
#     # C=np.cov(x_array,y_array)

#     # Soln to TLS: U.T*U*N=0, N!=0
#     # To solve homogenous eq:
#     M=np.dot(C.T,C)
#     N=np.dot(M.T,M)
#     print("N shape",N.shape)
#     eigvals, eigvecs = np.linalg.eig(N)
#     print("eigvals shape:",eigvals.shape)
#     soln_idx=np.argmin(eigvals)
#     soln_vec=eigvecs[:,soln_idx]
#     print("parameters shape:",soln_vec.shape)
#     print("TLS result:",soln_vec)
#     y_pred=soln_vec[0]*x_array + soln_vec[1]
#     # y_pred=400*np.array([0,50,100]) + soln_vec[1]

#     points=np.vstack((x_array,y_pred)).T
#     print(points.shape)

#     return points

def fit_total_least_squares(x_array,y_array):
    x_mean=np.mean(x_array)
    y_mean=np.mean(y_array)

    U=np.vstack((x_array-x_mean,y_array-y_mean)).T

    A = np.dot(U.T,U)

    # A*x=0, x!=0
    # soln is eigvec(A.T*A) corresponding to smallest eigval
    
    M = np.dot(A.T,A)

    eigvals,eigvecs=np.linalg.eig(M)
    print("eigvals: ",eigvals)
    print("eigvecs: ",eigvecs)
    soln_idx=np.argmin(eigvals)
    soln_vec=eigvecs[:,soln_idx]
    print("TLS params:",soln_vec)

    d = soln_vec[0] * x_mean + soln_vec[1] * y_mean
    # ax + by = d
    # y = (d - ax)/b
    y_pred = (d - soln_vec[0] * x_array)/soln_vec[1]
    # y_pred=soln_vec[0] * x_array + soln_vec[1]
    points=np.vstack((x_array,y_pred)).T
    # print("xshape:",x_array.shape)
    # print("yshape:",y_pred.shape)
    # print("points shape: ",points.shape)

    return points

def RANSAC_points(x,parameters):
    # print(len(parameters))
    y_pred=(-parameters[2] - parameters[0]*x)/parameters[1]
    return x,y_pred


def calculate_inliers(x_array,y_array,parameters,thresh):
    dist=np.abs(-parameters[0] * x_array - parameters[1] * y_array - parameters[2])/np.sqrt(parameters[0]**2 + parameters[1]**1)
    inlier=np.where(dist<thresh,1,0)
    num_inliers=np.sum(inlier)
    return num_inliers


def RANSAC_fit(x_array,y_array,p,e,s,thresh):
    N=int(log(1-p)/(log(1-np.power((1-e),s))))
    # print("iterations : ",N)
    best_parameters=[]
    max_inliers=0
    for i in range(N):
        rn1=random.randint(0,len(x_array)-1)
        rn2=random.randint(0,len(x_array)-1)
        x1,y1=x_array[rn1],y_array[rn1]
        x2,y2=x_array[rn2],y_array[rn2]

        # standard form of line: px + qy + r
        if(x1!=x2):
            p = (y2-y1)/(x1-x2)
        else:
            p=1e8
        q=1
        r=y1+p*x1
        parameters=[p,q,r]
        num_inliers=calculate_inliers(x_array,y_array,parameters,thresh)

        if(num_inliers>max_inliers):
            best_parameters=parameters
            max_inliers=num_inliers
        
    return best_parameters


def main():
    data=pd.read_csv('/home/nitesh/programming/ENPM673/HW1/src/data.csv')

    age=data['age']
    charges=data['charges']
    age=age.to_numpy(dtype=int)
    charges=charges.to_numpy(dtype=float)
    plt.scatter(age,charges)
    
    age_mean=np.mean(age)
    charges_mean=np.mean(charges)
    
    cov1=np.dot(age-age_mean,(charges-charges_mean).T)/(len(age)-1)
    # print("cov1: ",cov1)
    cov2=np.dot(charges-charges_mean,(age-age_mean).T)/(len(charges)-1)
    # print("cov2: ",cov2)
    
    age_var=np.dot(age-age_mean,(age-age_mean).T)/(len(age)-1)
    # print("age_var:",age_var)
    charges_var=np.dot(charges-charges_mean,(charges-charges_mean).T)/(len(age)-1)
    # print("charges_var:",charges_var)

    cov_mtx=np.array([[age_var,cov1],[cov2,charges_var]])
    eigvals,eigvecs=np.linalg.eig(cov_mtx)
    
    origin = np.array([[age_mean,age_mean],[charges_mean,charges_mean]])
    plt.quiver(*origin, eigvecs[:,0], eigvecs[:,1], color=['r','b'], scale=10)
    points=fit_linear_least_squares(age,charges)
    # plt.show()
    # print("y shape before:",y.shape)
    plt.plot(points[:,0],points[:,1],'r',label="standard least squares")

    points_TLS=fit_total_least_squares(age,charges)
    # print(points_TLS)
    plt.plot(points_TLS[:,0],points_TLS[:,1],'g',label='total least squares')

    # RANSAC parameters
    s=2 # line
    e=0.4
    p=0.9999
    thresh=10
    
    RANSAC_params=RANSAC_fit(age,charges,p,e,s,thresh)
    x_ransac,y_ransac=RANSAC_points(age,RANSAC_params)
    plt.plot(x_ransac,y_ransac,'b',label='ransac')
    plt.legend()
    plt.show()


if __name__=="__main__":
    main()
    
