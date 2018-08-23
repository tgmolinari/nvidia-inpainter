import torch
import torch.nn as nn
from model.partial_conv import NaivePConv2d

class inpainter(nn.Module):
    ''' The Liu et al. (2018) model'''
    def __init__(self):
        super(inpainter, self).__init__()
        self.lrlu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.hot = False

        # encoder
        self.pconv1 = NaivePConv2d(3,  64,  7, stride = 2, padding = 1, bias = True)
        self.pconv2 = NaivePConv2d(64, 128, 5, stride = 2, padding = 1)
        self.en_bn2 = nn.BatchNorm2d(128)
        self.pconv3 = NaivePConv2d(128, 256, 5, stride = 2, padding = 1)
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

        # upsampler, only need one since it's parameter free & all upsamples are factor 2 w/ nearest neighbor interpolation
        self.upsample = nn.modules.upsampling.Upsample(scale_factor = 2)

    def forward(self, inp, mask):
            x1, m1 = self.pconv1(inp, mask)
            x = self.relu(x1)
            print('pc 1: ' + str(x.size()) + '\tmask: ' + str(m1.size()))
            x2, m2 = self.pconv2(x, m1)
            x = self.en_bn2(x2)
            x = self.relu(x)
            print('pc 2: ' + str(x.size()) + '\tmask: ' + str(m2.size()))
            x3, m3 = self.pconv3(x, m2)
            x = self.en_bn3(x3)
            x = self.relu(x)
            print('pc 3: ' + str(x.size()) + '\tmask: ' + str(m3.size()))
            x4, m4 = self.pconv4(x, m3)
            x = self.en_bn4(x4)
            x = self.relu(x)        
            print('pc 4: ' + str(x.size()) + '\tmask: ' + str(m4.size()))
            x5, m5 = self.pconv5(x, m4)
            x = self.en_bn5(x5)
            x = self.relu(x)
            print('pc 5: ' + str(x.size()) + '\tmask: ' + str(m5.size()))
            x6, m6 = self.pconv6(x, m5)
            x = self.en_bn6(x6)
            x = self.relu(x)
            print('pc 6: ' + str(x.size()) + '\tmask: ' + str(m6.size()))
            x7, m7 = self.pconv7(x, m6)
            x = self.en_bn7(x7)
            print('pc 7: ' + str(x.size()) + '\tmask: ' + str(m7.size()))
            x = self.relu(x)
            x, m = self.pconv8(x, m7)
            x = self.en_bn8(x)
            x = self.relu(x)
            print('pc 8: ' + str(x.size()) + '\tmask: ' + str(m.size()))

            # decoder
            print('upsample after pc 8: ' + str(self.upsample(x).size()))
            x = torch.cat((self.upsample(x), x7), dim = 1)
            m = torch.cat((self.upsample(m), m7), dim = 1) 
            x, m = self.pconv9(x, m)
            x = self.de_bn9(x)
            x = self.lrlu(x)
            print('pc 9: ' + str(x.size()) + '\tmask: ' + str(m.size()))
            x = torch.cat((self.upsample(x), x6), dim = 1)
            m = torch.cat((self.upsample(m), m6), dim = 1)
            x, m = self.pconv10(x, m)
            x = self.de_bn10(x)
            x = self.lrlu(x)
            print('pc 10: ' + str(x.size()) + '\tmask: ' + str(m.size()))
            x = torch.cat((self.upsample(x), x5), dim = 1)
            m = torch.cat((self.upsample(m), m5), dim = 1)
            x, m = self.pconv11(x, m)
            print('pc 11: ' + str(x.size()) + '\tmask: ' + str(m.size()))
            x = self.de_bn11(x)
            x = self.lrlu(x)
            x = torch.cat((self.upsample(x), x4), dim = 1)
            m = torch.cat((self.upsample(m), m4), dim = 1)
            x, m = self.pconv12(x, m)
            x = self.de_bn12(x)
            x = self.lrlu(x)
            print('pc 12: ' + str(x.size()) + '\tmask: ' + str(m.size()))
            x = torch.cat((self.upsample(x), x3), dim = 1)
            m = torch.cat((self.upsample(m), m3), dim = 1)
            x, m = self.pconv13(x, m)
            x = self.de_bn13(x)
            x = self.lrlu(x)
            print('pc 103 ' + str(x.size()) + '\tmask: ' + str(m.size()))            
            x = torch.cat((self.upsample(x), x2), dim = 1)
            m = torch.cat((self.upsample(m), m2), dim = 1)
            x, m = self.pconv14(x, m)
            x = self.de_bn14(x)
            x = self.lrlu(x)
            print('pc 14: ' + str(x.size()) + '\tmask: ' + str(m.size()))
            x = torch.cat((self.upsample(x), x1), dim = 1)
            m = torch.cat((self.upsample(m), m1), dim = 1)
            x, m = self.pconv15(x, m)
            x = self.de_bn15(x)
            x = self.lrlu(x)
            print('pc 15: ' + str(x.size()) + '\tmask: ' + str(m.size()))
            x = torch.cat((self.upsample(x), inp), dim = 1)
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
                    child.track_running_stats = False # check this actually works after you flip the switch!
        else:
            self.hot = True
