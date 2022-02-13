#!/usr/bin/python3
import numpy as np
import math

def homography(x,y,xp,yp):
    A=np.array([[-x[0],-y[0],-1,0,0,0,x[0]*xp[0],y[0]*xp[0],xp[0]],
                [0,0,0,-x[0],y[0],-1,x[0]*yp[0],y[0]*yp[0],yp[0]],
                [-x[1],-y[1],-1,0,0,0,x[1]*xp[1],y[1]*xp[1],xp[1]],
                [0,0,0,-x[1],y[1],-1,x[1]*yp[1],y[1]*yp[1],yp[1]],
                [-x[2],-y[2],-1,0,0,0,x[2]*xp[2],y[2]*xp[2],xp[2]],
                [0,0,0,-x[2],y[2],-1,x[2]*yp[2],y[2]*yp[2],yp[2]],
                [-x[3],-y[3],-1,0,0,0,x[3]*xp[3],y[3]*xp[3],xp[3]],
                [0,0,0,-x[3],y[3],-1,x[3]*yp[3],y[3]*yp[3],yp[3]]])

    return A

def sort_eig_pairs(W,U):
    couple=[(W[i],U[:,i]) for i in range(len(W))]
    couple.sort(reverse=True)
    W=[couple[i][0] for i in range(len(couple))]
    U=[couple[i][1] for i in range(len(couple))]

    return np.array(W),np.array(U)

def main():
    x=np.array([5,150,150,5])
    y=np.array([5,5,150,150])
    xp=np.array([100,200,220,100])
    yp=np.array([100,80,80,200])

    A=homography(x,y,xp,yp) #mxn=8x9

    W1, U = np.linalg.eig(np.dot(A,A.T))

    W1,U=sort_eig_pairs(W1,U)

    W2,V=np.linalg.eig(np.dot(A.T,A))
    W2,V=sort_eig_pairs(W2,V)

    m,n=A.shape
    S = np.zeros((A.shape))
    for i in range(np.min(A.shape)):
        S[i,i] = np.sqrt(np.abs(W1[i]))

    A_pred = np.dot(np.dot(U, S), V.T)

    H=V.T[:,-1].reshape((3,3))
    print("Homography matrix is:")
    print(H)



if __name__=="__main__":
    main()