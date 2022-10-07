#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:22:39 2020

@author: yuanbi
"""

# import matplotlib.image as mpimg
import cv2
import os
import torch
import numpy as np
# import matplotlib.image as mpimg
# import torchvision
from torchvision import transforms
# import helper
from scipy.spatial.transform import Rotation
from platform import python_version
# import matplotlib.pyplot as plt

def get_traj_index(x,length):
    row_index=np.sum(np.array(length.cumsum()) <= x, axis=0)-1
    column_index=x-length.cumsum()[row_index]
    return (row_index,column_index)

def sample_uniform(original_num,sample_num):
    idx=np.round(np.linspace(0,original_num-1,sample_num)).astype(int)
    return idx.tolist()

def get_Pos(probe_pose):
    r=Rotation.from_quat(probe_pose[3:7])
    if python_version()[0]==2:
        R=r.as_dcm()
    else:
        R=r.as_matrix()
    a=np.reshape(probe_pose[0:3],(3,1))
    R=np.append(R,a,axis=1)
    R=np.append(R,np.array([[0,0,0,1]]),axis=0)
    p1=np.dot(R,np.transpose(np.array([0.0375/2,0,0,1])))
    p2=np.dot(R,np.transpose(np.array([-0.0375/2,0,0,1])))
    ang=np.dot(p1[0:2]-p2[0:2],np.transpose(np.array([1,0])))/np.linalg.norm(p1[0:2]-p2[0:2])
    ang=180-np.arccos(ang)*180/np.pi
    
    pos=[probe_pose[0],probe_pose[1],ang]
    return pos

class Dataset_loader(torch.utils.data.Dataset):
    def __init__(self, list_IDs, list_length, list_length_reduced, poses,NUM_DEMO,files,demo_path):
        'Initialization'
        self.list_IDs = list_IDs
        self.list_length=list_length
        self.list_length_reduced=list_length_reduced
#         self.ExpectedRanking=ExpectedRanking
        self.NUM_DEMO=NUM_DEMO
        self.files=files
        self.demo_path=demo_path
        self.poses=poses
        self.lookupTable=[]
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.23),
        ])
        
        for i in range(len(NUM_DEMO)):
            
            self.lookupTable.append(sample_uniform(int(self.list_length[i+1]),int(self.list_length_reduced[i+1])))
        
        end_poses=[]
        denom=[]
        for traj in poses:
            end_pos=get_Pos(traj[-1])
            end_poses.append(end_pos)
            temp_poses=[]
            for p in traj:
                pos=get_Pos(p)
                temp_poses.append(pos)
            temp_denom=np.amax(abs(np.array(temp_poses)-np.array(end_pos)),axis=0)
            print(temp_denom)
            denom.append(temp_denom)
                
        self.end_pose=np.mean(end_poses,axis=0)
        self.denom = np.amax(denom,axis=0)
        print(self.denom)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID_1, ID_2 = self.list_IDs[index]
        
        # Load data
        Demo_num_1,Frame_num_1=get_traj_index(ID_1,self.list_length_reduced)
        Demo_num_2,Frame_num_2=get_traj_index(ID_2,self.list_length_reduced)

        
#         Frame_num_1=(Frame_num_1+1)*self.factor+self.list_length[Demo_num_1+1]%self.factor-1
#         Frame_num_2=(Frame_num_2+1)*self.factor+self.list_length[Demo_num_2+1]%self.factor-1
        Frame_num_1=int(self.lookupTable[Demo_num_1][Frame_num_1])
        Frame_num_2=int(self.lookupTable[Demo_num_2][Frame_num_2])
        
        pos_1=np.array(get_Pos(self.poses[Demo_num_1][Frame_num_1]))
        end_pos_1=np.array(get_Pos(self.poses[Demo_num_1][-1]))
        Dis_1=np.linalg.norm(abs(pos_1-end_pos_1)/np.array(self.denom))
        
        pos_2=np.array(get_Pos(self.poses[Demo_num_2][Frame_num_2]))
        end_pos_2=np.array(get_Pos(self.poses[Demo_num_2][-1]))
        Dis_2=np.linalg.norm(abs(pos_2-end_pos_2)/np.array(self.denom))
        
        if Dis_2<Dis_1:
            X_2 = cv2.imread(self.demo_path+'/'
                               +str(self.NUM_DEMO[Demo_num_1])+'/'+self.files[Demo_num_1][Frame_num_1])
            X_1 = cv2.imread(self.demo_path+'/'
                               +str(self.NUM_DEMO[Demo_num_2])+'/'+self.files[Demo_num_2][Frame_num_2])
        else:
            X_1 = cv2.imread(self.demo_path+'/'
                               +str(self.NUM_DEMO[Demo_num_1])+'/'+self.files[Demo_num_1][Frame_num_1])
            X_2 = cv2.imread(self.demo_path+'/'
                               +str(self.NUM_DEMO[Demo_num_2])+'/'+self.files[Demo_num_2][Frame_num_2])
        X_1 = cv2.resize(X_1, (256,256),interpolation=cv2.INTER_LANCZOS4)
        X_2 = cv2.resize(X_2, (256,256),interpolation=cv2.INTER_LANCZOS4)
        
        X_1 = cv2.cvtColor(X_1, cv2.COLOR_BGR2GRAY)
        X_2 = cv2.cvtColor(X_2, cv2.COLOR_BGR2GRAY)

        # X_1_aug = X_1/255
        # X_2_aug = X_2/255 

        X_1_aug=self.img_augmentation(X_1)/255
        X_2_aug=self.img_augmentation(X_2)/255
        
        return [X_1_aug,X_2_aug]
    
    def img_augmentation(self,img):
        return self.contrast(self.brightness(self.noise_level(self.blurriness(self.sharpness(self.crop(img))))))
        
    def sharpness(self, img):
        alpha=np.random.rand()*20+10 #[10,30]
        prob=np.random.rand()
        if prob<0.1:
            blur = cv2.GaussianBlur(img,(0,0),1.0)
            blurr = cv2.GaussianBlur(blur,(0,0),1.0)
            unsharp_image = cv2.addWeighted(blur, alpha+1, blurr, -alpha, 0)
            return unsharp_image.astype('uint8')
        else:
            return img
    
    def blurriness(self, img):
        alpha=np.random.rand()*1.25+0.25 #[0.25,1.5]
        prob=np.random.rand()
        if prob<0.1:
            blur_image = cv2.GaussianBlur(img,(0,0),alpha)
            return blur_image.astype('uint8')
        else:
            return img
    
    def noise_level(self, img):
        alpha=np.random.rand()*0.04+0.01 #[0.01,0.05]
        prob=np.random.rand()
        if prob<0.1:
            gaussian = np.random.normal(0, alpha, (img.shape[0],img.shape[1]))*255
            noised_image=img+gaussian
            noised_image[noised_image>255]=255
            noised_image[noised_image<0]=0
            return noised_image.astype('uint8')
        else:
            return img
    
    def brightness(self, img):
        alpha=np.random.rand()*0.2-0.1
        alpha=int(alpha*255)
        prob=np.random.rand()
        if prob<0.1:
            brightness_image=img+alpha
            brightness_image[brightness_image>255] = 255
            brightness_image[brightness_image<0] = 0
            return brightness_image.astype('uint8')
        else:
            return img
        
    
    def contrast(self, img):
        alpha=np.random.rand()*2.5+0.5
        prob=np.random.rand()
        if prob<0.1:
            invGamma = 1 / alpha
            table = [((i / 255) ** invGamma) * 255 for i in range(256)]
            table = np.array(table, np.uint8)
    
            contrast_image=cv2.LUT(img, table)
    
            return contrast_image.astype('uint8')
        else:
            return img
        
    def crop(self, img):
        alpha=np.random.rand()*0.1+0.8
        prob=np.random.rand()
        if prob<0.5:
            height=img.shape[0]
            
            croped_height=int(height*alpha)
            
            prob_ = np.random.rand()
            if prob_<0.34:
                mid = 0
            elif prob_<0.67:
                mid = int(height*(1-alpha))
            else:
                mid = int(height*(1-alpha)//2)
                
            # mid = int(height*(1-alpha))
            
            # mid=np.random.randint(int(height*(1-alpha)//2),int(height*(1-alpha)))
            
            image_croped = img[mid:croped_height+mid,:]
            
            image_croped = cv2.resize(image_croped, (256,256),interpolation=cv2.INTER_LANCZOS4)
            
            return image_croped
        else:
            return img
    
    def flip(self, img):
        alpha=np.random.rand()
        prob=np.random.rand()
        if prob<0.1:
            image_flipped = cv2.flip(img,1)
            # if alpha<0.33:
            #     image_flipped=cv2.flip(img, 0)
            # elif alpha<0.66:
            #     image_flipped=cv2.flip(img, 1)
            # else:
            #     image_flipped=cv2.flip(img, -1)
            return image_flipped
        else:
            return img

    
    