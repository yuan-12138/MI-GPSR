#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:22:04 2020

@author: yuanbi
"""

import argparse
import torch
import torch.optim as  optim
import torch.nn as nn
from torchvision.utils import save_image
from MI_Reward_Net import Recon_encoder, Mine, Recon_decoder, Reward_FC,\
    Recon_encoder_fusion, Recon_decoder_fusion, Reward_encoder_new
from utils import prepare_data, prepare_test_data
import os
from os import listdir
import shutil
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import random
import cv2
import matplotlib.pyplot as plt
import pickle


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

parser = argparse.ArgumentParser(description='MI_Reward_Net training')
parser.add_argument('--result_dir', type=str, default='./Recon_Netresult', metavar='DIR',
					help='output directory')
parser.add_argument('--save_dir', type=str, default='./ckpt', metavar='DIR',
					help='model saving directory')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
					help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
					help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: None')
parser.add_argument('--test_every', default=1, type=int, metavar='N',
					help='test after every epochs')
parser.add_argument('--num_worker', type=int, default=0, metavar='N',
					help='num_worker')

# model options
parser.add_argument('--lr', type=float, default=1e-5,
					help='learning rate')
parser.add_argument('--z_dim', type=int, default=32, metavar='N',
					help='latent vector size of encoder')
parser.add_argument('--input_dim', type=int, default=256 * 256, metavar='N',
					help='input dimension')
parser.add_argument('--input_channel', type=int, default=1, metavar='N',
					help='input channel')

args = parser.parse_args()

# kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}

pre_trained = False

NUM_DEMO=['1','3','5','7','9']
# NUM_DEMO=['1','2','3','4','5']
NUM_TEST_DEMO=['8']
discount_factor=4
discount_factor_test=4
demo_path="/home/robotics-meta/Project/yuanBi/Demonstrations/test_set_2/kidney/imageSet"
test_demo_path="/home/robotics-meta/Project/yuanBi/Demonstrations/test_set_2/kidney/imageSet"
pose_path="/home/robotics-meta/Project/yuanBi/Demonstrations/test_set_2/kidney/poseSet"
pretrain_path="./ckpt/8/checkpoint.pth"

max_grad_norm=10

Mine = Mine(args.z_dim).to(device)
optimizer_mine = optim.Adam(Mine.parameters(), lr=1e-4)
ma_rate=0.001

# Reward_Net = MI_Reward_Network(in_channels=args.input_channel, z_dim=args.z_dim).to(device)
# optimizer = optim.Adam(Reward_Net.parameters(), lr=args.lr)

Rec_encoder = Recon_encoder_fusion(args.input_channel,args.z_dim,init_features=64).to(device)
optimizer_rec_en = optim.Adam(Rec_encoder.parameters(), lr=args.lr)

Rec_decoder = Recon_decoder_fusion(args.input_channel,args.z_dim,init_features=64).to(device)
optimizer_rec_de = optim.Adam(Rec_decoder.parameters(), lr=args.lr)

Reward_encoder = Reward_encoder_new(args.input_channel,args.z_dim,init_features=64).to(device)
optimizer_reward_en = optim.Adam(Reward_encoder.parameters(), lr=args.lr)

Reward_FC = Reward_FC(args.z_dim).to(device)
optimizer_reward_fc = optim.Adam(Reward_FC.parameters(), lr=args.lr)

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

transform_image=transforms.Normalize(0.5,0.5)

sigmoid=nn.Sigmoid()

trainable_params_reward_en = sum(
	p.numel() for p in Reward_encoder.parameters() if p.requires_grad
)
trainable_params_reward_fc = sum(
	p.numel() for p in Reward_FC.parameters() if p.requires_grad
)
trainable_params_rec_en = sum(
	p.numel() for p in Rec_encoder.parameters() if p.requires_grad
)
trainable_params_rec_de = sum(
	p.numel() for p in Rec_decoder.parameters() if p.requires_grad
)

print('Reward encoder size:', trainable_params_reward_en, 'Reward FC size:', trainable_params_reward_fc)
print('Rec encoder size:', trainable_params_rec_en, 'Rec decoder size:', trainable_params_rec_de)

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
    # rec_loss = F.mse_loss(Recon_result, input, reduction='mean')
    
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

def update_Ads(input,train):
    z = Rec_encoder(input)
    r = Reward_FC(z)
    
    ads_loss = torch.sum(r)
    
    if train:
        ads_loss.backward()
        
        optimizer_rec_en.step()
        optimizer_reward_fc.step()
        
        reset_grad()
    
    return ads_loss

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

# def learn_mine(input):
#     with torch.no_grad():
#         z_a = Reward_encoder(input)
#         z_d = Rec_encoder(input)
       
#         z_d_shuffle = torch.index_select(z_d, 0, torch.randperm(z_d.shape[0]).to(device))
    
#     mi, t, et = mi_estimator(z_a, z_d, z_d_shuffle)
    
#     loss=-mi

#     loss.backward()
#     optimizer_mine.step()
#     reset_grad()
    
    
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
    # min_length=np.min(length[1:-1])
    length_reduced=[0]
    for i in NUM_DEMO:
        length_reduced.append(100)
    length_reduced=np.array(length_reduced)
    
    
    print(length)
    print(length_reduced)
#     print('calculating GroundTruthRanking')
#     GroundTruthRanking=ExpectedRanking(length_reduced)
    
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
    
    
    
    trainloader = prepare_data(length_reduced, poses, length,NUM_DEMO,files_training, demo_path, args.batch_size, args.num_worker)
    testloader = prepare_test_data(length_reduced_test, poses_test, length_test,NUM_TEST_DEMO,files_test, test_demo_path, args.batch_size, args.num_worker)

    counter=0
    
    training_loss_records = []
    test_loss_records = []
    
	# training
    for epoch in range(start_epoch, args.epochs):
        train_avg_seg_loss=0
        for i, data in enumerate(trainloader):
            
            input_1=data[0].view(-1,1,256,256).float().to(device)
            
            input_2=data[1].view(-1,1,256,256).float().to(device)
            
            for _ in range(1):
                learn_mine(input_1)
                learn_mine(input_2)

            recon_loss_1, Recon_result_1 = update_Rec(input_1,True)
            recon_loss_2, Recon_result_2 = update_Rec(input_2,True)
            recon_loss = (recon_loss_1+recon_loss_2)*0.5
            
            pc_loss, reward_1, reward_2 = update_Reward(input_1,input_2,True)
            
            mi_loss_1 = update_MI(input_1,True)
            mi_loss_2 = update_MI(input_2,True)
            mi_loss = 0.5*(mi_loss_1+mi_loss_2)
            
            # ads_loss_1 = update_Ads(input_1,True)
            # ads_loss_2 = update_Ads(input_2,True)
            # constant_loss = ads_loss_1+ads_loss_2
            
            # loss = recon_loss+pc_loss+mi_loss+constant_loss
            loss = recon_loss+pc_loss+mi_loss
            train_avg_seg_loss+=loss.item()
            training_loss_records.append(loss.item())
            if (i + 1) % 20 == 0:
                # print("\r Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}, Mutual loss: {:.4f} Recon loss {:.4f} PC loss {:.4f} Constant loss {:.4f} Reward_1 {:.4f} Reward_2 {:.4f}".format(epoch + 1, args.epochs, i + 1, len(trainloader), 
                #     loss.item(), mi_loss.item(), recon_loss.item(), pc_loss.item(), 
                #     constant_loss.item(), reward_1[0].item(), reward_2[0].item()), end='')
                print("\r Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}, Mutual loss: {:.4f} Recon loss {:.4f} PC loss {:.4f} Reward_1 {:.4f} Reward_2 {:.4f}"\
                      .format(epoch + 1, args.epochs, i + 1, len(trainloader), 
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
    				}, False, args.save_dir)
                    return
                
                # training_loss_records.append([loss.item(), recon_loss.item(), pc_loss.item(), mi_loss.item()])

            # if i == 0:
            #     x_concat = torch.cat([input_1.view(-1, 1, 256, 256), Recon_result_1.view(-1, 1, 256, 256), input_2.view(-1, 1, 256, 256), Recon_result_2.view(-1, 1, 256, 256)], dim=3)
            #     save_image(x_concat, ("./%s/reconstructed-%d.png" % (args.result_dir, epoch + 1)))
            
            if (i + 1) % 500 == 0:
                counter+=1
                x_concat = torch.cat([input_1.view(-1, 1, 256, 256), Recon_result_1.view(-1, 1, 256, 256), input_2.view(-1, 1, 256, 256), Recon_result_2.view(-1, 1, 256, 256)], dim=3)
                save_image(x_concat, ("./%s/reconstructed-%d.png" % (args.result_dir, counter)))
                
                test_avg_loss = 0.0
                test_avg_mutual_loss = 0.0
                with torch.no_grad():
                    
                    for idx, test_data in enumerate(testloader):
    					# get the inputs; data is a list of [inputs, labels]
                        
                        test_input_1=data[0].view(-1,1,256,256).float().to(device)
                        test_input_2=data[1].view(-1,1,256,256).float().to(device)
                        
                        recon_loss_1, Recon_result_1 = update_Rec(test_input_1,False)
                        recon_loss_2, Recon_result_2 = update_Rec(test_input_2,False)
                        recon_loss = (recon_loss_1+recon_loss_2)*0.5
                        
                        pc_loss, reward_1, reward_2 = update_Reward(test_input_1,test_input_2,False)
                        
                        mi_loss_1 = update_MI(test_input_1,False)
                        mi_loss_2 = update_MI(test_input_2,False)
                        test_mi_loss = 0.5*(mi_loss_1+mi_loss_2)
                        
                        # ads_loss_1 = update_Ads(test_input_1,False)
                        # ads_loss_2 = update_Ads(test_input_2,False)
                        # constant_loss = ads_loss_1+ads_loss_2
                        
                        # test_loss = recon_loss+pc_loss+test_mi_loss+constant_loss
                        test_loss = pc_loss
                        
                        # test_mutual_loss,test_z_a,test_z_a_1=mutual_information_loss(test_inputs,test_labels)
    
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
    				}, is_best, args.save_dir)
                    
                    test_loss_records.append(test_avg_loss.item())
                    np.savetxt(os.path.join(args.save_dir,'training_loss_record.txt'), training_loss_records, fmt='%f')
                    np.savetxt(os.path.join(args.save_dir,'test_loss_record.txt'), test_loss_records, fmt='%f')
                    # training_loss_records.append(train_avg_seg_loss)
                    # train_avg_seg_loss = 0
                    
                    # if len(test_loss_records)==2:
                    #     plt.clf()
                    #     plt.ylabel('test_loss')
                    #     plt.xlabel('Training progress #')
                    #     plt.plot([len(test_loss_records)-1,len(test_loss_records)], [test_loss_records[-2],test_loss_records[-1]],'b', alpha=0.3)
                    #     plt.plot([len(training_loss_records)-1,len(training_loss_records)], [training_loss_records[-2],training_loss_records[-1]],'g', alpha=0.3)
                    #     plt.pause(0.05)
                    # elif len(test_loss_records)>2:
                    #     plt.plot([len(test_loss_records)-1,len(test_loss_records)], [test_loss_records[-2],test_loss_records[-1]],'b', alpha=0.3)
                    #     plt.plot([len(training_loss_records)-1,len(training_loss_records)], [training_loss_records[-2],training_loss_records[-1]],'g', alpha=0.3)
                    #     plt.pause(0.05)
            
		# testing
#         if (epoch + 1) % args.test_every == 0:
            
#             test_avg_loss = 0.0
#             test_avg_mutual_loss = 0.0
#             with torch.no_grad():
                
#                 for idx, test_data in enumerate(testloader):
# 					# get the inputs; data is a list of [inputs, labels]
                    
#                     test_input_1=data[0].view(-1,1,256,256).float().to(device)
#                     test_input_2=data[1].view(-1,1,256,256).float().to(device)
                    
#                     test_loss, _, _, _, test_mi_loss, _, _, _, _=loss_func(test_input_1, test_input_2)
                    
#                     # test_mutual_loss,test_z_a,test_z_a_1=mutual_information_loss(test_inputs,test_labels)

#                     test_avg_loss += test_loss
#                     test_avg_mutual_loss += torch.abs(test_mi_loss)

#                 test_avg_loss /= len(testloader.dataset)
#                 test_avg_mutual_loss /= len(testloader.dataset)
#                 print("Average Test loss {:.4f} Average Test Mutual loss {:.4f}".format(test_avg_loss.item(), test_avg_mutual_loss))

# 				# we randomly sample some images' latent vectors from its distribution
#                 # z = torch.randn(args.batch_size, args.z_dim).to(device)
#                 # random_res = myVAE.decode(z).view(-1, 1, 256, 256)
#                 # save_image(random_res, "./%s/random_sampled-%d.png" % (args.result_dir, epoch + 1))

# 				# save model
#                 is_best = test_avg_loss < best_test_loss
#                 best_test_loss = min(test_avg_loss, best_test_loss)
#                 save_checkpoint({
# 					'epoch': epoch,
# 					'best_test_loss': best_test_loss,
# 					'state_dict': Reward_Net.state_dict(),
#                     'state_dict_mine': Mine.state_dict(),
#                     'state_dict_reward': Reward_Net.Reward_Net.state_dict(),
# 					'optimizer': optimizer.state_dict(),
#                     'optimizer_mine': optimizer_mine.state_dict(),
# 				}, is_best, args.save_dir)



if __name__ == '__main__':
	train()
