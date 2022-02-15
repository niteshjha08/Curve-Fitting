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
    # print("LLS params:",res)
    # points=res[0]*age_augmented[:,0]+res[0]*age_augmented[:,1]
    points=get_best_fit_line(age,res,degree=1)

    # return age,points
    return points


def fit_total_least_squares(x_array,y_array):
    x_mean=np.mean(x_array)
    y_mean=np.mean(y_array)
    

    U=np.vstack((x_array-x_mean,y_array-y_mean)).T

    A = np.dot(U.T,U)

    # A*x=0, x!=0
    # soln is eigvec(A.T*A) corresponding to smallest eigval
    
    M = np.dot(A.T,A)

    eigvals,eigvecs=np.linalg.eig(M)

    soln_idx=np.argmin(eigvals)
    soln_vec=eigvecs[:,soln_idx]


    d = soln_vec[0] * x_mean + soln_vec[1] * y_mean
    # ax + by = d
    # y = (d - ax)/b
    y_pred = (d - soln_vec[0] * x_array)/soln_vec[1]

    points=np.vstack((x_array,y_pred)).T

    return points

def RANSAC_points(x,parameters):
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

        # standard form of line: px + qy + r = 0
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

def get_eigen_vectors(age,charges):
    age=age/np.max(age)
    charges=charges/np.max(charges)
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
    print(cov_mtx)
    eigvals,eigvecs=np.linalg.eig(cov_mtx)
    print(eigvecs)
    return eigvals,eigvecs
    

def main():
    data=pd.read_csv('./data.csv')

    age=data['age']
    charges=data['charges']
    age=age.to_numpy(dtype=int)
    charges=charges.to_numpy(dtype=float)
    # age=age/np.max(age)
    # charges=charges/np.max(charges)
    
    
    eigvals,eigvecs = get_eigen_vectors(age,charges)
    age_mean=np.mean(age)
    charges_mean=np.mean(charges)

    plt.scatter(age,charges)
    print(eigvecs)
    # origin = np.array([[age_mean,age_mean],[charges_mean,charges_mean]])
    origin = np.array([age_mean,charges_mean])
    e2=eigvecs[:,1]
    e1=eigvecs[:,0]
    plt.quiver(*origin, *e2, color='b', scale=10)
    plt.quiver(*origin, *e1, color='r', scale=10)

    # plt.show()
    points=fit_linear_least_squares(age,charges)
    
    plt.plot(points[:,0],points[:,1],'r',label="standard least squares")
    # plt.show()
    points_TLS=fit_total_least_squares(age,charges)
    
    plt.plot(points_TLS[:,0],points_TLS[:,1],'g',label='total least squares')
    # plt.show()

    # RANSAC parameters
    s=2 # line
    e=0.4
    p=0.99999
    thresh=10
    
    RANSAC_params=RANSAC_fit(age,charges,p,e,s,thresh)
    x_ransac,y_ransac=RANSAC_points(age,RANSAC_params)
    plt.plot(x_ransac,y_ransac,'b',label='RANSAC')
    plt.legend()
    plt.show()


if __name__=="__main__":
    main()
    
