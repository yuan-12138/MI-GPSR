#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:22:04 2020

@author: yuanbi
"""

import torch
import torch.optim as  optim
import torch.nn as nn
from torchvision.utils import save_image
from MI_Reward_Net import Mine, Reward_FC,\
    Recon_encoder_fusion, Recon_decoder_fusion, Reward_encoder_new
from utils import prepare_data, prepare_test_data
import os
from os import listdir
import shutil
import numpy as np
import torch.nn.functional as F
import pickle


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

result_dir = './Recon_Netresult'
save_dir = './ckpt'
batch_size = 12
epochs = 2
seed = 1
test_every = 40
num_worker = 0

lr = 1e-5
lr_mine = 1e-4
z_dim = 32
input_dim = 256 * 256
input_channel = 1

pre_trained = False

NUM_DEMO=['1','2','3','4','5']
NUM_TEST_DEMO=['10']

demo_path="./Demonstrations/imageSet"
test_demo_path="./Demonstrations/imageSet"
pose_path="./Demonstrations/poseSet"
pretrain_path=""

max_grad_norm=10

Mine = Mine(z_dim).to(device)
optimizer_mine = optim.Adam(Mine.parameters(), lr=lr_mine)
ma_rate=0.001

Rec_encoder = Recon_encoder_fusion(input_channel,z_dim,init_features=64).to(device)
optimizer_rec_en = optim.Adam(Rec_encoder.parameters(), lr=lr)

Rec_decoder = Recon_decoder_fusion(input_channel,z_dim,init_features=64).to(device)
optimizer_rec_de = optim.Adam(Rec_decoder.parameters(), lr=lr)

Reward_encoder = Reward_encoder_new(input_channel,z_dim,init_features=64).to(device)
optimizer_reward_en = optim.Adam(Reward_encoder.parameters(), lr=lr)

Reward_FC = Reward_FC(z_dim).to(device)
optimizer_reward_fc = optim.Adam(Reward_FC.parameters(), lr=lr)

if pre_trained:
    pretrain_para=torch.load(pretrain_path,map_location=device)
    Mine.load_state_dict(pretrain_para['state_dict_mine'])
    Rec_encoder.load_state_dict(pretrain_para['state_dict_rec_en'])
    Rec_decoder.load_state_dict(pretrain_para['state_dict_rec_de'])
    Reward_encoder.load_state_dict(pretrain_para['state_dict_reward_en'])
    Reward_FC.load_state_dict(pretrain_para['state_dict_reward_fc'])
    
    optimizer_mine.load_state_dict(pretrain_para['optimizer_mine'])
    optimizer_rec_en.load_state_dict(pretrain_para['optimizer_rec_en'])
    optimizer_rec_de.load_state_dict(pretrain_para['optimizer_rec_de'])
    optimizer_reward_en.load_state_dict(pretrain_para['optimizer_reward_en'])
    optimizer_reward_fc.load_state_dict(pretrain_para['optimizer_reward_fc'])
    del pretrain_para
    torch.cuda.empty_cache()


sigmoid=nn.Sigmoid()


def save_checkpoint(state, is_best, outdir):

	if not os.path.exists(outdir):
		os.makedirs(outdir)

	checkpoint_file = os.path.join(outdir, 'checkpoint.pth')
	best_file = os.path.join(outdir, 'model_best.pth')
	torch.save(state, checkpoint_file)
	if is_best:
		shutil.copyfile(checkpoint_file, best_file)


def reset_grad():
    optimizer_mine.zero_grad()
    optimizer_rec_en.zero_grad()
    optimizer_rec_de.zero_grad()
    optimizer_reward_en.zero_grad()
    optimizer_reward_fc.zero_grad()

def update_Rec(input,train):
    z_a = Reward_encoder(input)
    z_d = Rec_encoder(input)
    
    Recon_result = Rec_decoder(z_d,z_a)
    
    rec_loss = F.mse_loss(Recon_result, input, reduction='mean')*(len(input))*2
    
    if train:
        rec_loss.backward()
        
        optimizer_rec_en.step()
        optimizer_rec_de.step()
        optimizer_reward_en.step()
        
        reset_grad()
    
    return rec_loss, Recon_result

def update_Reward(input_1,input_2,train):
    z_1 = Reward_encoder(input_1)
    z_2 = Reward_encoder(input_2)
    
    reward_1 = Reward_FC(z_1)
    reward_2 = Reward_FC(z_2)
    
    pc_loss = torch.sum(sigmoid(10*(reward_2-reward_1)))
    
    if train:
        pc_loss.backward()
        
        optimizer_reward_en.step()
        optimizer_reward_fc.step()
        
        reset_grad()
    
    return pc_loss, reward_1, reward_2


def update_MI(input,train):
    z_a = Reward_encoder(input)
    z_d = Rec_encoder(input)
    
    z_d_shuffle = torch.index_select(z_d, 0, torch.randperm(z_d.shape[0]).to(device))
    
    mutual_loss,_,_ = mi_estimator(z_a, z_d, z_d_shuffle)
    
    mutual_loss=F.leaky_relu(mutual_loss)
    
    if train:
        mutual_loss.backward()
        
        optimizer_rec_en.step()
        optimizer_reward_en.step()
        
        reset_grad()
    
    return mutual_loss
    

def mi_estimator(x, y, y_):
    joint, marginal = Mine(x, y), Mine(x, y_)
    return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal))), joint, marginal
    
    
def learn_mine(input, ma_rate=0.001):
    
    with torch.no_grad():
        z_a = Reward_encoder(input)
        z_d = Rec_encoder(input)
       
        z_d_shuffle = torch.index_select(z_d, 0, torch.randperm(z_d.shape[0]).to(device))
    
    et = torch.mean(torch.exp(Mine(z_a, z_d_shuffle)))
    if Mine.ma_et is None:
        Mine.ma_et = et.detach().item()
    Mine.ma_et += ma_rate * (et.detach().item() - Mine.ma_et)
    mutual_information = torch.mean(Mine(z_a, z_d)) - torch.log(et) * et.detach() /Mine.ma_et
    loss = - 10*mutual_information
    
    loss.backward()
    optimizer_mine.step()
    reset_grad()
    
    return mutual_information.item()


def train():
    

    start_epoch = 0
    best_test_loss = np.finfo('f').max
    
    poses=[]
    length=[0]
    files_training=[]
    for i in NUM_DEMO:
        f_demo = listdir(demo_path+"/"+str(i))
        f_demo.sort()
        files_training.append(f_demo)
        length.append(len(f_demo))
        f_pose=listdir(pose_path+"/"+str(i))
        pose=pickle.load(open(pose_path+"/"+str(i)+'/'+f_pose[0], 'rb'))
        poses.append(pose)
    length=np.array(length)
    length_reduced=[0]
    for i in NUM_DEMO:
        length_reduced.append(100)
    length_reduced=np.array(length_reduced)
    
    
    print(length)
    print(length_reduced)
    
    # load test demonstrations
    poses_test=[]
    length_test=[0]
    files_test=[]
    for i in NUM_TEST_DEMO:
        f = listdir(test_demo_path+"/"+str(i))
        f.sort()
        files_test.append(f)
        length_test.append(len(f))
        f_pose_test=listdir(pose_path+"/"+str(i))
        pose_test=pickle.load(open(pose_path+"/"+str(i)+'/'+f_pose_test[0], 'rb'))
        poses_test.append(pose_test)
    length_test=np.array(length_test)
    length_reduced_test=[0]
    for i in NUM_TEST_DEMO:
        length_reduced_test.append(100)
    length_reduced_test=np.array(length_reduced_test)
    
    
    
    trainloader = prepare_data(length_reduced, poses, length,NUM_DEMO,files_training, demo_path, batch_size, num_worker)
    testloader = prepare_test_data(length_reduced_test, poses_test, length_test,NUM_TEST_DEMO,files_test, test_demo_path, batch_size, num_worker)

    counter=0
    
    training_loss_records = []
    test_loss_records = []
    
	# training
    for epoch in range(start_epoch, epochs):
        train_avg_seg_loss=0
        for i, data in enumerate(trainloader):
            
            input_1=data[0].view(-1,1,256,256).float().to(device)
            
            input_2=data[1].view(-1,1,256,256).float().to(device)
            
            for _ in range(5):
                learn_mine(input_1)
                learn_mine(input_2)

            recon_loss_1, Recon_result_1 = update_Rec(input_1,True)
            recon_loss_2, Recon_result_2 = update_Rec(input_2,True)
            recon_loss = (recon_loss_1+recon_loss_2)*0.5
            
            pc_loss, reward_1, reward_2 = update_Reward(input_1,input_2,True)
            
            mi_loss_1 = update_MI(input_1,True)
            mi_loss_2 = update_MI(input_2,True)
            mi_loss = 0.5*(mi_loss_1+mi_loss_2)
            
            loss = recon_loss+pc_loss+mi_loss
            train_avg_seg_loss+=loss.item()
            training_loss_records.append(loss.item())
            if (i + 1) % 20 == 0:
                print("\r Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}, Mutual loss: {:.4f} Recon loss {:.4f} PC loss {:.4f} Reward_1 {:.4f} Reward_2 {:.4f}"\
                      .format(epoch + 1, epochs, i + 1, len(trainloader), 
                    loss.item(), mi_loss.item(), recon_loss.item(), pc_loss.item(), 
                    reward_1[0].item(), reward_2[0].item()), end='')
                if torch.any(torch.isnan(loss)):
                    save_checkpoint({
    					'epoch': epoch,
    					'best_test_loss': 0,
    					'state_dict_rec_en': Rec_encoder.state_dict(),
                        'state_dict_rec_de': Rec_decoder.state_dict(),
                        'state_dict_reward_en': Reward_encoder.state_dict(),
                        'state_dict_reward_fc': Reward_FC.state_dict(),
                        'state_dict_mine': Mine.state_dict(),
    					'optimizer_rec_en': optimizer_rec_en.state_dict(),
                        'optimizer_rec_de': optimizer_rec_de.state_dict(),
                        'optimizer_reward_en': optimizer_reward_en.state_dict(),
                        'optimizer_reward_fc': optimizer_reward_fc.state_dict(),
                        'optimizer_mine': optimizer_mine.state_dict(),
    				}, False, save_dir)
                    return
            
            if (i + 1) % test_every == 0:
                counter+=1
                x_concat = torch.cat([input_1.view(-1, 1, 256, 256), Recon_result_1.view(-1, 1, 256, 256), input_2.view(-1, 1, 256, 256), Recon_result_2.view(-1, 1, 256, 256)], dim=3)
                save_image(x_concat, ("./%s/reconstructed-%d.png" % (result_dir, counter)))
                
                test_avg_loss = 0.0
                test_avg_mutual_loss = 0.0
                with torch.no_grad():
                    
                    for idx, test_data in enumerate(testloader):                        
                        test_input_1=data[0].view(-1,1,256,256).float().to(device)
                        test_input_2=data[1].view(-1,1,256,256).float().to(device)
                        
                        recon_loss_1, Recon_result_1 = update_Rec(test_input_1,False)
                        recon_loss_2, Recon_result_2 = update_Rec(test_input_2,False)
                        recon_loss = (recon_loss_1+recon_loss_2)*0.5
                        
                        pc_loss, reward_1, reward_2 = update_Reward(test_input_1,test_input_2,False)
                        
                        mi_loss_1 = update_MI(test_input_1,False)
                        mi_loss_2 = update_MI(test_input_2,False)
                        test_mi_loss = 0.5*(mi_loss_1+mi_loss_2)

                        test_loss = pc_loss
                            
                        test_avg_loss += test_loss
                        test_avg_mutual_loss += torch.abs(test_mi_loss)
    
                    test_avg_loss /= (idx+1)
                    test_avg_mutual_loss /= (idx+1)
                    train_avg_seg_loss /= 200
                    print("Average Test loss {:.4f} Average Test Mutual loss {:.4f}".format(test_avg_loss.item(), test_avg_mutual_loss))
    
    				# save model
                    is_best = test_avg_loss < best_test_loss
                    best_test_loss = min(test_avg_loss, best_test_loss)
                    save_checkpoint({
    					'epoch': epoch,
    					'best_test_loss': best_test_loss,
    					'state_dict_rec_en': Rec_encoder.state_dict(),
                        'state_dict_rec_de': Rec_decoder.state_dict(),
                        'state_dict_reward_en': Reward_encoder.state_dict(),
                        'state_dict_reward_fc': Reward_FC.state_dict(),
                        'state_dict_mine': Mine.state_dict(),
    					'optimizer_rec_en': optimizer_rec_en.state_dict(),
                        'optimizer_rec_de': optimizer_rec_de.state_dict(),
                        'optimizer_reward_en': optimizer_reward_en.state_dict(),
                        'optimizer_reward_fc': optimizer_reward_fc.state_dict(),
                        'optimizer_mine': optimizer_mine.state_dict(),
    				}, is_best, save_dir)
                    
                    test_loss_records.append(test_avg_loss.item())
                    np.savetxt(os.path.join(save_dir,'training_loss_record.txt'), training_loss_records, fmt='%f')
                    np.savetxt(os.path.join(save_dir,'test_loss_record.txt'), test_loss_records, fmt='%f')




if __name__ == '__main__':
	train()
