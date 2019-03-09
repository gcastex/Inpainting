import numpy as np
import h5py
import torch
import torch.utils.data
from torch import nn
import matplotlib.pylab as plt
from random import randint

from my_lib.gc_nn_lib import *
from my_lib.gc_autoencoder import *
from gc_in1 import *

torch.cuda.is_available()
#dir_sim = '/mnt/home/fvillaescusa/data/Giullame_density_fields/'

data_ts = get_basic_im_data()
test = data_ts[50,:,:]
#plot_2d_field(torch.log(test))

#plot_2d_field(data_ts[50,:,:])

data_test = data_ts[270,:,:]
rdict = generate_2dhole(data_test)
#data_test = data_test.view(1,1,512,512)
test = rdict["hdata"]
mask = rdict["mask"]
#dpix = [x for x  in range(data_ts.view(data_ts.size(0),-1).size(1)) if not x in hpix]

#train_data, norm_data = generate_training_data(data_ts[:250,:,:],hpix, dpix)
train_data, norm_data = generate_training_data(data_ts[:250,:,:], mask)

#ndpix = 512*512
#ndpix = len(dpix)
#nhpix = len(hpix)

#plot_2d_field(torch.log(test[270,:,:]))

class stupidnn(nn.Module):
    def __init__(self):
        super(stupidnn, self).__init__()
        self.neck = nn.Sequential(
            nn.Linear(128*128, 100), nn.ReLU(True),
            nn.Linear(100, 100), nn.ReLU(True),
            nn.Linear(100, 100), nn.ReLU(True),
            nn.Linear(100, 128*128))
    def forward(self, x):
        nel = x.size(0)
        nch = x.size(1)
        nside = x.size(2)
        x = x.view(nel,-1)
        x = self.neck(x)
        x = x.view(nel, nch, nside, -1)
	return x

def init_uni(m):
    if type(m) == nn.Conv2d:
        nn.init.uniform_(m.weight)

num_epochs = 100
batch_size = 10

#model = UNet() #autoencoder()
model = autoencoder_cba()
#model = stupidnn()
model=model.cuda()
#model.encoder.apply(init_uni)
#model.decoder.apply(init_uni)
criterion = nn.MSELoss()

learning_rate = 1e-3
optimizer = torch.optim.Adam(
#optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate)#, momentum=)

dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
#tensor_data: pixels known
#tensor_target: pixels to inpaint
#dataset = data_utils.TensorDataset(tensor_data, tensor_target)

## Train model
##############

loss_model = np.zeros([num_epochs,1])

model.train()
for epoch in range(num_epochs):
	#Test
#    outmod = norm_data(model(norm_data(data_test).cuda()).cpu(), back = True)
#    plot_2d_field(outmod.view(512,-1).detach(), save=('figs/epochs/model_%s.png')%(epoch), clim=clim)
    for data in dataloader:
        img, targets = data
        #img = img.view(img.size(0), -1)
        #img = Variable(img).cuda()
        # ===================forward=====================
	img = img.cuda()
	targets = targets.cuda()
        output = model(img)
        loss = criterion(output, targets)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    #outmod = norm_data(model(norm_data(data_test).cuda()).cpu(), back = True)
    #plot_2d_field(outmod.view(512,-1).detach())
    loss_model[epoch] = loss.data[0]
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))

    #if epoch % 10 == 0:
    #    pic = to_img(output.cpu().data)
    #    save_image(pic, './mlp_img/image_{}.png'.format(epoch))
##############
#plt.plot(loss_model, 'r')
#plt.show()

model.eval()



def plot_results(k, save='', clim=[0,0]):
        clim = [-10,10]
        pixside = data_ts.size(2)
        data_test = (data_ts[k,:,:]*mask).view(1,1,pixside,-1)
        outmod = norm_data(model(norm_data(data_test).cuda()).cpu(), back = True)
        f, axarr = plt.subplots(1,2)
        axarr[0].set_title('Original')
        axarr[1].set_title('Model')
        imgplot2 = axarr[1].pcolormesh(outmod.view(pixside,-1).detach().numpy(), edgecolors='None', cmap='jet')
        #axarr[1].imshow(outmod.view(pixside,-1).detach().numpy(), origin='lower', cmap='jet')
        #axarr[0].imshow((data_ts[k,:,:]*mask).numpy(), origin='lower', cmap='jet')
        imgplot1 = axarr[0].pcolormesh((data_ts[k,:,:]*mask).numpy(), edgecolors='None', cmap='jet')
        #plt.title('Density Field')
        #axarr[0].colorbar()
        imgplot1.set_clim(clim[0],clim[1])
        imgplot2.set_clim(clim[0],clim[1])
        f.colorbar(imgplot1)
        if save != '':
                #if clim !=[0,0]:
                #        plt.clim(clim[0],clim[1])
                plt.savefig(save)
                plt.clf()
        else:
                plt.show()



plot_results(300)

plot_2d_field(outmod.view(pixside,-1).detach(), save='figs/model.png')
plot_2d_field(data_ts[k,:,:], save='figs/original.png')
#plot_results(dpix, data_test[dpix], hpix, filled)

#device = 0
#torch.cuda.get_device_properties(device).total_memory
#torch.cuda.get_device_name(0)
#model.encoder[0].weight

d=nn.Sequential(
             Interpolate(scale_factor = 2, mode='bilinear'),
             nn.ReflectionPad2d(1),
             nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0))

d2=nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)

a=next(iter(dataloader))
plot_2d_field(a[1][2,0,:,:])

## Save figures of inpainting examples using model

modelname = 'autoencoder_cba'
for k in range(1,10):
	plot_results(k,save=('figs/model_%s_%s.png')%(modelname,k))
	plt.close('all')

for k in range(300,320):
        plot_results(mask, k,save=('figs/model_%s_%s.png')%(modelname,k))
        plt.close('all')



