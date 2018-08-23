import torch
from torch import nn
from partial_conv import NaivePConv2d

class inpainter(nn.Module):
    ''' The Liu et al. (2018) model'''
    def __init__(self):
        super(inpainter, self).__init__()
        self.lrlu = nn.LeakyReLU(0.2)
        self.hot = False

        # encoder
        self.pconv1 = NaivePConv2d(3,  64,  7, stride = 2, bias = True)
        self.pconv2 = NaivePConv2d(64, 128, 5, stride = 2)
        self.en_bn2 = nn.BatchNorm2d(64)
        self.pconv3 = NaivePConv2d(128, 256, 5, stride = 2)
        self.en_bn3 = nn.BatchNorm2d(128)
        self.pconv4 = NaivePConv2d(256, 512, 3, stride = 2)
        self.en_bn4 = nn.BatchNorm2d(256)
        self.pconv5 = NaivePConv2d(512, 512, 3, stride = 2)
        self.en_bn5 = nn.BatchNorm2d(512)
        self.pconv6 = NaivePConv2d(512, 512, 3, stride = 2)
        self.en_bn6 = nn.BatchNorm2d(512)
        self.pconv7 = NaivePConv2d(512, 512, 3, stride = 2)
        self.en_bn7 = nn.BatchNorm2d(512)
        self.pconv8 = NaivePConv2d(512, 512, 3, stride = 2)
        self.en_bn8 = nn.BatchNorm2d(512)


        # decoder
        self.pconv9 = NaivePConv2d(512,512,3)
        self.de_bn9 = nn.BatchNorm2d(512)
        self.pconv10 = NaivePConv2d(512,512,3)
        self.de_bn10 = nn.BatchNorm2d(512)
        self.pconv11 = NaivePConv2d(512,512,3)
        self.de_bn11 = nn.BatchNorm2d(512)
        self.pconv12 = NaivePConv2d(512,512,3)
        self.de_bn12 = nn.BatchNorm2d(512)
        self.pconv13 = NaivePConv2d(512,256,3)
        self.de_bn13 = nn.BatchNorm2d(512)
        self.pconv14 = NaivePConv2d(256,128,3)
        self.de_bn14 = nn.BatchNorm2d(256)
        self.pconv15 = NaivePConv2d(128,64,3)
        self.de_bn15 = nn.BatchNorm2d(128)
        self.pconv16 = NaivePConv2d(64,3,3, bias = True)

        # upsampler, only need one since it's parameter free & all upsamples are factor 2 w/ nearest neighbor interpolation
        self.upsample = nn.modules.upsampling.Upsample(scale_factor = 2)

    def forward(self, inp, mask):
            x1, m1 = self.pconv1(inp, mask)
            x = nn.ReLU(x1)
            x2, m2 = self.pconv2(x, m)
            x = self.bn2(x2)
            x = nn.ReLU(x)
            x3, m3 = self.pconv3(x, m)
            x = self.bn3(x3)
            x = nn.ReLU(x)
            x4, m4 = self.pconv4(x, m)
            x = self.bn4(x4)
            x = nn.ReLU(x)        
            x5, m5 = self.pconv5(x, m)
            x = self.bn5(x5)
            x = nn.ReLU(x)
            x6, m6 = self.pconv6(x, m)
            x = self.bn6(x6)
            x = nn.ReLU(x)
            x7, m7 = self.pconv7(x, m)
            x = self.bn7(x7)
            x = nn.ReLU(x)
            x, m = self.pconv8(x, m)
            x = self.bn8(x)
            x = nn.ReLU(x)
            
            # decoder
            x = torch.cat((self.upsample(x), x7), dim = 1)
            m = torch.cat((self.upsample(m), m7), dim = 1)
            x, m = self.pconv9(x, m)
            x = self.bn9(x)
            x = self.lrlu(x)
            x = torch.cat((self.upsample(x), x6), dim = 1)
            m = torch.cat((self.upsample(m), m6), dim = 1)
            x, m = self.pconv10(x, m)
            x = self.bn10(x)
            x = self.lrlu(x)
            x = torch.cat((self.upsample(x), x5), dim = 1)
            m = torch.cat((self.upsample(m), m5), dim = 1)
            x, m = self.pconv11(x, m)
            x = self.bn11(x)
            x = self.lrlu(x)
            x = torch.cat((self.upsample(x), x4), dim = 1)
            m = torch.cat((self.upsample(m), m4), dim = 1)
            x, m = self.pconv12(x, m)
            x = self.bn12(x)
            x = self.lrlu(x)
            x = torch.cat((self.upsample(x), x3), dim = 1)
            m = torch.cat((self.upsample(m), m3), dim = 1)
            x, m = self.pconv13(x, m)
            x = self.bn13(x)
            x = self.lrlu(x)
            x = torch.cat((self.upsample(x), x2), dim = 1)
            m = torch.cat((self.upsample(m), m2), dim = 1)
            x, m = self.pconv14(x, m)
            x = self.bn14(x)
            x = self.lrlu(x)
            x = torch.cat((self.upsample(x), x1), dim = 1)
            m = torch.cat((self.upsample(m), m1), dim = 1)
            x, m = self.pconv15(x, m)
            x = self.bn15(x)
            x = self.lrlu(x)
            x = torch.cat((self.upsample(x), inp))
            x, m = self.pconv16(x, mask)

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
                    #nn.BatchNorm.params(FREEZE EM!)
        else:
            self.hot = True

