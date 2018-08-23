import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from model.partial_conv import NaivePConv2d

class inpainter(nn.Module):
    ''' The Liu et al. (2018) model'''
    def __init__(self):
        super(inpainter, self).__init__()
        self.lrlu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.hot = False

        # encoder
        self.pconv1 = NaivePConv2d(3,  64,  7, stride = 2, padding = 3, bias = True)
        self.pconv2 = NaivePConv2d(64, 128, 5, stride = 2, padding = 2)
        self.en_bn2 = nn.BatchNorm2d(128)
        self.pconv3 = NaivePConv2d(128, 256, 5, stride = 2, padding = 2)
        self.en_bn3 = nn.BatchNorm2d(256)
        self.pconv4 = NaivePConv2d(256, 512, 3, stride = 2, padding = 1)
        self.en_bn4 = nn.BatchNorm2d(512)
        self.pconv5 = NaivePConv2d(512, 512, 3, stride = 2, padding = 1)
        self.en_bn5 = nn.BatchNorm2d(512)
        self.pconv6 = NaivePConv2d(512, 512, 3, stride = 2, padding = 1)
        self.en_bn6 = nn.BatchNorm2d(512)
        self.pconv7 = NaivePConv2d(512, 512, 3, stride = 2, padding = 1)
        self.en_bn7 = nn.BatchNorm2d(512)
        self.pconv8 = NaivePConv2d(512, 512, 3, stride = 2, padding = 1)
        self.en_bn8 = nn.BatchNorm2d(512)


        # decoder, input channels following paper architecture convention
        self.pconv9 = NaivePConv2d(512 + 512, 512, 3, padding = 1)
        self.de_bn9 = nn.BatchNorm2d(512)
        self.pconv10 = NaivePConv2d(512 + 512, 512, 3, padding = 1)
        self.de_bn10 = nn.BatchNorm2d(512)
        self.pconv11 = NaivePConv2d(512 + 512, 512, 3, padding = 1)
        self.de_bn11 = nn.BatchNorm2d(512)
        self.pconv12 = NaivePConv2d(512 + 512, 512, 3, padding = 1)
        self.de_bn12 = nn.BatchNorm2d(512)
        self.pconv13 = NaivePConv2d(512 + 256, 256, 3, padding = 1)
        self.de_bn13 = nn.BatchNorm2d(256)
        self.pconv14 = NaivePConv2d(256 + 128, 128, 3, padding = 1)
        self.de_bn14 = nn.BatchNorm2d(128)
        self.pconv15 = NaivePConv2d(128 + 64, 64, 3, padding = 1)
        self.de_bn15 = nn.BatchNorm2d(64)
        self.pconv16 = NaivePConv2d(64 + 3, 3, 3, padding = 1, bias = True)


    def forward(self, inp, mask):
            x1, m1 = self.pconv1(inp, mask)
            x = self.relu(x1)
            x2, m2 = self.pconv2(x, m1)
            x = self.en_bn2(x2)
            x = self.relu(x)
            x3, m3 = self.pconv3(x, m2)
            x = self.en_bn3(x3)
            x = self.relu(x)
            x4, m4 = self.pconv4(x, m3)
            x = self.en_bn4(x4)
            x = self.relu(x)        
            x5, m5 = self.pconv5(x, m4)
            x = self.en_bn5(x5)
            x = self.relu(x)
            x6, m6 = self.pconv6(x, m5)
            x = self.en_bn6(x6)
            x = self.relu(x)
            x7, m7 = self.pconv7(x, m6)
            x = self.en_bn7(x7)
            x = self.relu(x)
            x, m = self.pconv8(x, m7)
            x = self.en_bn8(x)
            x = self.relu(x)

            # decoder
            x = torch.cat((interpolate(x, scale_factor = 2), x7), dim = 1)
            m = torch.cat((interpolate(m, scale_factor = 2), m7), dim = 1) 
            x, m = self.pconv9(x, m)
            x = self.de_bn9(x)
            x = self.lrlu(x)
            x = torch.cat((interpolate(x, scale_factor = 2), x6), dim = 1)
            m = torch.cat((interpolate(m, scale_factor = 2), m6), dim = 1)
            x, m = self.pconv10(x, m)
            x = self.de_bn10(x)
            x = self.lrlu(x)
            x = torch.cat((interpolate(x, scale_factor = 2), x5), dim = 1)
            m = torch.cat((interpolate(m, scale_factor = 2), m5), dim = 1)
            x, m = self.pconv11(x, m)
            x = self.de_bn11(x)
            x = self.lrlu(x)
            x = torch.cat((interpolate(x, scale_factor = 2), x4), dim = 1)
            m = torch.cat((interpolate(m, scale_factor = 2), m4), dim = 1)
            x, m = self.pconv12(x, m)
            x = self.de_bn12(x)
            x = self.lrlu(x)
            x = torch.cat((interpolate(x, scale_factor = 2), x3), dim = 1)
            m = torch.cat((interpolate(m, scale_factor = 2), m3), dim = 1)
            x, m = self.pconv13(x, m)
            x = self.de_bn13(x)
            x = self.lrlu(x)
            x = torch.cat((interpolate(x, scale_factor = 2), x2), dim = 1)
            m = torch.cat((interpolate(m, scale_factor = 2), m2), dim = 1)
            x, m = self.pconv14(x, m)
            x = self.de_bn14(x)
            x = self.lrlu(x)
            x = torch.cat((interpolate(x, scale_factor = 2), x1), dim = 1)
            m = torch.cat((interpolate(m, scale_factor = 2), m1), dim = 1)
            x, m = self.pconv15(x, m)
            x = self.de_bn15(x)
            x = self.lrlu(x)
            x = torch.cat((interpolate(x, scale_factor = 2), inp), dim = 1)
            m = torch.cat((interpolate(m, scale_factor = 2), mask), dim = 1)
            x, m = self.pconv16(x, m)
            del x7,x6,x5,x4,x3,x2,x1
            del m7,m6,m5,m4,m3,m2,m1,m           
 
            return x

    def init_weights(self):
        '''kaiming_normal the weights for each p conv layer'''
        for name, child in self.named_children():
            if 'pconv' in name:
                nn.init.kaiming_normal_(child.weight)
        return None

    def toggle_hot_start(self):
        if self.hot:
            self.hot = False
            for name, child in self.named_children():
                if 'en_bn' in name:
                    child.track_running_stats = False # check this actually works after you flip the switch!
        else:
            self.hot = True
