import torch.nn as nn
    
class netG(nn.Module):
    def __init__(self, nc = 1, nz = 1, ngf = 32, gfs = 4, ngpu = 1):
        super(netG, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
                
                nn.ConvTranspose2d(in_channels = nz, out_channels = ngf*8, kernel_size = gfs, stride = 2, padding = 1, dilation=1, bias = True), 
                nn.ReLU(True),
                nn.InstanceNorm2d(ngf * 8),
                
                nn.ConvTranspose2d(in_channels = ngf*8, out_channels = ngf*4, kernel_size = gfs, stride = 2, padding = 1, dilation=1, bias = True), 
                nn.ReLU(True),
                nn.InstanceNorm2d(ngf * 4),

                nn.ConvTranspose2d(in_channels = ngf*4, out_channels = ngf*2, kernel_size = gfs, stride = 2, padding = 1, dilation=1, bias = True),
                nn.ReLU(True),
                nn.InstanceNorm2d(ngf*2),
                
                nn.ConvTranspose2d(in_channels = ngf*2, out_channels = ngf, kernel_size = gfs, stride = 2, padding = 1, dilation=1, bias = True),
                nn.ReLU(True),
                nn.InstanceNorm2d(ngf),
               
                nn.ConvTranspose2d(in_channels = ngf, out_channels = nc, kernel_size = gfs, stride = 2, padding = 1, dilation=1, bias = True),  
#                nn.Tanh()
                nn.Sigmoid()
                
            )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)
        return output


class netD(nn.Module):
    def __init__(self, nc = 1, ndf = 32, dfs = 4, ngpu = 1):
        super(netD, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(

            nn.Conv2d(in_channels = 1, out_channels = ndf, kernel_size = dfs, stride = 2, padding = 1, dilation=1, bias = True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(ndf),
            
            nn.Conv2d(in_channels = ndf, out_channels = ndf*2, kernel_size = dfs, stride = 2, padding = 1, dilation=1, bias = True),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(ndf*2),

            nn.Conv2d(in_channels = ndf*2, out_channels = ndf*4, kernel_size = dfs, stride = 2, padding = 1, dilation=1, bias = True),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(ndf*4),

            nn.Conv2d(in_channels = ndf*4, out_channels = ndf*8, kernel_size = dfs, stride = 2, padding = 1, dilation=1, bias = True),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(ndf*8),
            
            nn.Conv2d(in_channels = ndf*8, out_channels = nc, kernel_size = dfs, stride = 2, padding = 1, dilation=1, bias = True)
        )
        self.main = main
    
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)
       
        return output.view(-1, 1).squeeze(1)
