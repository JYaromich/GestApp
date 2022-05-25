from torch import nn
import torch
from torch.nn import functional as F
import albumentations as alb


def get_transformer(image_size: tuple) -> 'transformers':
    return alb.Compose([
            alb.Resize(height=image_size[0], width=image_size[1]),
            alb.ToGray(p=1),
        ])

LABEL2INDX = {
    '01_palm': 0,
    '02_l': 1,
    '03_fist': 2,
    '04_fist_moved': 3,
    '05_thumb': 4,
    '06_index': 5,
    '07_ok': 6,
    '08_palm_moved': 7,
    '09_c': 8,
    '10_down': 9
}

INDX2LABEL = {indx: label for label, indx in LABEL2INDX.items()}


class GestRecogBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size=(3, 3), pool_size=2, drop_out=0.3):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_features, 
                              out_channels=int(out_features / 2), 
                              kernel_size=kernel_size, padding='same')
        
        self.conv_2 = nn.Conv2d(in_channels=int(out_features / 2), 
                              out_channels=out_features, 
                              kernel_size=kernel_size, padding='same')
        
        torch.nn.init.kaiming_normal_(self.conv_1.weight)
        torch.nn.init.kaiming_normal_(self.conv_2.weight)

        self.max_pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        self.bn = nn.BatchNorm2d(num_features=out_features)
        self.drop_out = nn.Dropout2d(drop_out)
        

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.bn(x)
        x = self.max_pool(x)
        return x

class GestRecogNN(torch.nn.Module):
    def get_dense_block(self, input_features, out_features, p_dropout):
        return nn.Sequential(
            nn.Linear(input_features, out_features), 
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(p_dropout)
        )
            
    def __init__(self, in_features, out_features):
        super().__init__()
        self.base_block = GestRecogBlock(in_features=in_features, out_features=64)
        self.base_block_2 = GestRecogBlock(in_features=64, out_features=128)
        self.base_block_3 = GestRecogBlock(in_features=128, out_features=256)
        self.base_block_4 = GestRecogBlock(in_features=256, out_features=512)
        self.classifier = nn.Sequential(
            self.get_dense_block(input_features=100352, out_features=256, p_dropout=0.2),
            self.get_dense_block(input_features=256, out_features=128, p_dropout=0.2),
            nn.Linear(in_features=128, out_features=out_features), 
            nn.ReLU())
        
    def forward(self, x):
        x = self.base_block(x)
        x = self.base_block_2(x)
        x = self.base_block_3(x)
        x = self.base_block_4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x