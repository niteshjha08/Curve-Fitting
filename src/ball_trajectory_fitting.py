#!/usr/bin/python3
import cv2
import numpy as np
import matplotlib.pyplot as plt

def standard_least_squares(x_array,y_array):
    p1=np.dot(x_array.T,x_array)
    # print(p1.shape)
    p2=np.linalg.inv(p1)
    p=np.dot(p2,x_array.T)
    x=np.dot(p,y_array)

    return x 

def get_best_fit_line(x_array,parameters,degree):
    x_array=x_array.astype(int)
    y_array=np.zeros(len(x_array))

    for i in range(degree+1): 
        temp=parameters[i]*(x_array**(degree-i))
        y_array+=temp 
   
    points=np.vstack((x_array,y_array)).T
    points=points.astype(int)

    return points

def main():
    smooth_trajectory_path='/home/nitesh/programming/ENPM673/HW1/videos/ball_video1.mp4'
    noise_trajectory_path='/home/nitesh/programming/ENPM673/HW1/videos/ball_video2.mp4'

    # Replace path for desired video result
    cap=cv2.VideoCapture(noise_trajectory_path)

    cv2.namedWindow('img',cv2.WINDOW_FREERATIO)
    cv2.namedWindow('red_mask',cv2.WINDOW_FREERATIO)
    cv2.namedWindow('points_img',cv2.WINDOW_FREERATIO)

    x_array=[]
    y_array=[]

    ret,img=cap.read()
    points_img=np.ones(img.shape)*255
    while(cv2.waitKey(5)!=ord('q')):

        ret,img=cap.read()
        if ret:
          
            red_mask=np.where((img[:,:,2]>150) & (img[:,:,0]<150),img[:,:,2],0)
            red_mask_indices=np.argwhere((img[:,:,2]>150) & (img[:,:,0]<150))
   
            ball_x_max=np.max(red_mask_indices[:,1])
            ball_x_min=np.min(red_mask_indices[:,1])
            ball_y_min=np.max(red_mask_indices[:,0])
            ball_y_max=np.min(red_mask_indices[:,0])
            ball_center=int((ball_x_min+ball_x_max)/2)

            x_array.append([ball_center**2,ball_center,1])
            x_array.append([ball_center**2,ball_center,1])
            y_array.extend([ball_y_max,ball_y_min])

            cv2.circle(img,(ball_center,ball_y_max),5,(255,0,0),-1)
            cv2.circle(img,(ball_center,ball_y_min),5,(255,255,0),-1)

            cv2.circle(points_img,(ball_center,ball_y_max),10,(255,0,0),-1)
            cv2.circle(points_img,(ball_center,ball_y_min),10,(255,255,0),-1)
            
            cv2.imshow('img',img)
            cv2.imshow('red_mask',red_mask)
            cv2.imshow('points_img',points_img)  
            # cv2.waitKey(0)
        else: 
            break
            # pass
    x_array=np.array(x_array)
    y_array=np.array(y_array)

    res=standard_least_squares(x_array,y_array)
    print(x_array[:,1].shape)
    print(y_array.shape)
    # print("res123:",res)
    points=get_best_fit_line(x_array[:,1],res,degree=2)
    points_img=cv2.polylines(points_img,[points],False,(0,255,0),10)
    cv2.imshow('points_img',points_img)
    cv2.waitKey()


if __name__=="__main__":
   main()
 
    