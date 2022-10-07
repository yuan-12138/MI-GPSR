import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


class Unflatten(nn.Module):
	def __init__(self, channel, height, width):
		super(Unflatten, self).__init__()
		self.channel = channel
		self.height = height
		self.width = width

	def forward(self, input):
		return input.view(input.size(0), self.channel, self.height, self.width)

class DoubleConv2d(nn.Module):
    def __init__(self,in_channels,features):
        super(DoubleConv2d, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=int(features/16), num_channels=features),
                nn.ReLU(inplace=True)
                # nn.Conv2d(features, features, kernel_size=3, padding=1),
                # nn.GroupNorm(num_groups=int(features/16), num_channels=features),
                # nn.ReLU(inplace=True)
                )

    def forward(self,x):
        return self.block(x)

class DoubleConv2d_s2(nn.Module):
    def __init__(self,in_channels,features):
        super(DoubleConv2d_s2, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups=int(features/16), num_channels=features),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups=int(features/16), num_channels=features),
                nn.ReLU(inplace=True)
                )

    def forward(self,x):
        return self.block(x)
    
class DoubleUpConv2d_s2(nn.Module):
    def __init__(self,in_channels,features):
        super(DoubleUpConv2d_s2, self).__init__()
        self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, features, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups=int(features/16), num_channels=features),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(features, features, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups=int(features/16), num_channels=features),
                nn.ReLU(inplace=True)
                )

    def forward(self,x):
        return self.block(x)
    
class Mine(nn.Module):
    def __init__(self,input_dim):
        super(Mine, self).__init__()
        
        self.ma_et=None
        
        self.fc1_x = nn.Linear(input_dim, 512)
        self.fc1_y = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,1)
        
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0.0)
                    
    def forward(self, x,y):
        h1 = F.leaky_relu(self.fc1_x(x)+self.fc1_y(y))
        h2 = F.leaky_relu(self.fc2(h1))
        h3 = F.leaky_relu(self.fc3(h2))
        return h3
    
    
class Reward_encoder(nn.Module):
    def __init__(self,in_channels=1,z_dim=512,init_features=64):
        super(Reward_encoder, self).__init__()
        
        features = init_features
        
        self.encoder_1 = DoubleConv2d_s2(in_channels,4*features)
        self.pool_1 = nn.MaxPool2d(2, 2)
        
        self.encoder_2 = DoubleConv2d_s2(4*features,16*features)
        self.pool_2 = nn.MaxPool2d(2, 2)
        
        self.bottleneck = DoubleConv2d(16*features,16*features)
        
        self.Flatten = Flatten()        
        
        self.fc = nn.Sequential(
            nn.Linear(4*4*1024, 256*16),
            nn.ReLU(),
            nn.Linear(256*16, z_dim),
            nn.Tanh()
        )
    def forward(self, input):
        enc_1 = self.encoder_1(input)
        
        enc_2 = self.encoder_2(self.pool_1(enc_1))
        
        bottleneck = self.bottleneck(self.pool_2(enc_2))
        
        z = self.fc(self.Flatten(bottleneck))
        
        return z
    
class Reward_FC(nn.Module):
    def __init__(self,z_dim=512):
        super(Reward_FC, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self, z):
        r = self.fc(z)
        
        return r
        
class Recon_encoder_fusion(nn.Module):
    def __init__(self,in_channels=1,z_dim=512,init_features=64):
        super(Recon_encoder_fusion, self).__init__()
        
        features = init_features
        
        self.encoder = nn.Sequential(
 			nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
 			nn.Conv2d(features, 2*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            # nn.Conv2d(2*features, 2*features, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(2*features, 2*features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*features, 2*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(2*features, 4*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*features, 4*features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*features, 4*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(4*features, 8*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*features, 8*features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*features, 8*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(8*features, 16*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features), num_channels=16*features),
            # nn.BatchNorm2d(16*features),
            nn.ReLU(inplace=True),

 			Flatten(),
            nn.Linear(16*features*8*8, 512),
            nn.ReLU(),
            nn.Linear(512, z_dim),
            nn.Tanh()
		)

    def forward(self, input):
        
        z = self.encoder(input)
        
        return z
    
class Recon_decoder_fusion(nn.Module):
    def __init__(self,in_channels=1,z_dim=512,init_features=64):
        super(Recon_decoder_fusion, self).__init__()
        
        features = init_features
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 16*features*8*8),
            nn.ReLU(),
            Unflatten(16*features, 8, 8),
            nn.ReLU(),
            # nn.ConvTranspose2d(16*features, 8*features, kernel_size=4, stride=4),
            nn.ConvTranspose2d(16*features, 8*features, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*features, 8*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(8*features, 4*features, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*features, 4*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(4*features, 2*features, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*features, 2*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(2*features, features, kernel_size=4, stride=4),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, in_channels, kernel_size=1),
            nn.Sigmoid()
		)

        
    def forward(self, z_a, z_d):
        
        Recon_result = self.decoder(z_a+z_d)
        
        return Recon_result
    
    
class Reward_encoder_new(nn.Module):
    def __init__(self,in_channels=1,z_dim=512,init_features=64):
        super(Reward_encoder_new, self).__init__()
        
        features = init_features
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
 			nn.Conv2d(features, 2*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            # nn.Conv2d(2*features, 2*features, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(2*features, 2*features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*features, 2*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(2*features, 4*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*features, 4*features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*features, 4*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(4*features, 8*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*features, 8*features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*features, 8*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(8*features, 16*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features), num_channels=16*features),
            # nn.BatchNorm2d(16*features),
            nn.ReLU(inplace=True),

 			Flatten(),
            nn.Linear(16*features*8*8, 512),
            nn.ReLU(),
            nn.Linear(512, z_dim),
            nn.Tanh()
            )
        
    def forward(self, input):
        
        return self.encoder(input)

