## Functions to generate dummy dataset, basic holes in images, visualisation of results.
import numpy as np
import h5py
import torch
import torch.utils.data
from torch import nn
import matplotlib.pylab as plt
from random import randint

def get_basic_im_data():
	dir_sim = '/mnt/home/gcastex/Soft/Simulations_P/paco_density_fields/'
	filename = dir_sim+'df_10_z=0.hdf5'
	f = h5py.File(filename, 'r')
	# List all groups
	#print("Keys: %s" % f.keys())
	a_group_key = list(f.keys())[0]
	# Get the data
	data = list(f[a_group_key])
	data_np = np.array(data)
	data_ts = torch.log(torch.from_numpy(data_np)+1)
	### Override with simple tests
	data_ts = data_ts[:,:128,:128]*0.
	pixside = data_ts.size(2)
	index = torch.randperm(512)
	for ii in range(256):
		i = index[ii]
		j = randint(round(pixside*.5), round(pixside*.8))
		data_ts[i,:,:]=randint(-10,10)
		data_ts[i,:,:j]=randint(-10,10)
		k = randint(round(pixside*.2), round(pixside*.5))
		data_ts[i,:,k:j]=randint(-10,10)
	for ii in range(256,512):
		i = index[ii]
		j = randint(round(pixside*.5), round(pixside*.8))
		data_ts[i,:,:]=randint(-10,10)
		data_ts[i,:j,:]=randint(-10,10)
		k = randint(round(pixside*.2), round(pixside*.5))
		data_ts[i,k:j,:]=randint(-10,10)
	return data_ts


def generate_hole(data):
        hdata = data.clone()
        hdata[250:300,250:300,250:300]=-1
        #pixels = [i for i, x in enumerate(hdata) if x ==-1]
        return hdata# {"hdata":hdata, "pixels": pixels}


def generate_2dhole(d2data):
        hdata = d2data.clone()
        #h_pixels = list(range(250,300))
        #w_pixels = list(range(250,300))
        pixside = hdata.size(1)
        hmin = round(pixside*.4)
        hmax = round(pixside*.6)
        h_pixels = list(range(hmin,hmax))
        w_pixels = list(range(hmin,hmax))
        mask = torch.ones(hdata.size(0),hdata.size(1))
        for i in h_pixels:
                for j in w_pixels:
                        mask[i, j]=0
                        hdata[i, j]=-1
        #pixels = [i for i, x in enumerate(hdata.view(-1)) if x ==-1]
        return {"hdata":hdata, "mask": mask}

def generate_training_data(data, mask):
        train_data = data.clone()
        #train_data = train_data.view(train_data.size(0),-1)
        #targets = data.view(data.size(0),-1)
        #pixels = [x for x  in range(tdata.size(1)) if not x in pixels_to_fill]
        #train_data[:, pixels_to_fill]=0        
        targets = train_data.clone()
        for i in range(train_data.size(0)):
                train_data[i,:,:] = train_data[i,:,:]*mask
        targets = targets.view(train_data.size(0),1,train_data.size(1),-1)
        train_data = train_data.view(train_data.size(0),1,train_data.size(1),-1)
        # normalization
        mu = train_data.mean()
        sigma = train_data.std()
        def norm_data(x, back = False, getmu = False, getsigma = False):
                if getmu:
                        return mu
                if getsigma:
                        return sigma
                if back:
                        return (x*sigma)+mu
                else:
                        return (x-mu)/sigma
        train_data = norm_data(train_data)
        targets = norm_data(targets)
        train = torch.utils.data.TensorDataset(train_data, targets)
        return train, norm_data

def generate_hole(data):
        hdata = data.clone()
        hdata[250:300,250:300,250:300]=-1
        #pixels = [i for i, x in enumerate(hdata) if x ==-1]
        return hdata# {"hdata":hdata, "pixels": pixels}


def generate_2dhole(d2data):
        hdata = d2data.clone()
        #h_pixels = list(range(250,300))
        #w_pixels = list(range(250,300))
        pixside = hdata.size(1)
        hmin = round(pixside*.4)
        hmax = round(pixside*.6)
        h_pixels = list(range(hmin,hmax))
        w_pixels = list(range(hmin,hmax))
        mask = torch.ones(hdata.size(0),hdata.size(1))
        for i in h_pixels:
                for j in w_pixels:
                        mask[i, j]=0
                        hdata[i, j]=-1
        #pixels = [i for i, x in enumerate(hdata.view(-1)) if x ==-1]
        return {"hdata":hdata, "mask": mask}


def generate_training_data(data, mask):
        train_data = data.clone()
        #train_data = train_data.view(train_data.size(0),-1)
        #targets = data.view(data.size(0),-1)
        #pixels = [x for x  in range(tdata.size(1)) if not x in pixels_to_fill]
        #train_data[:, pixels_to_fill]=0        
        targets = train_data.clone()
        for i in range(train_data.size(0)):
                train_data[i,:,:] = train_data[i,:,:]*mask
        targets = targets.view(train_data.size(0),1,train_data.size(1),-1)
        train_data = train_data.view(train_data.size(0),1,train_data.size(1),-1)
        # normalization
        mu = train_data.mean()
        sigma = train_data.std()
        def norm_data(x, back = False, getmu = False, getsigma = False):
                if getmu:
                        return mu
                if getsigma:
                        return sigma
                if back:
                        return (x*sigma)+mu
                else:
                        return (x-mu)/sigma
        train_data = norm_data(train_data)
        targets = norm_data(targets)
        train = torch.utils.data.TensorDataset(train_data, targets)
        return train, norm_data




