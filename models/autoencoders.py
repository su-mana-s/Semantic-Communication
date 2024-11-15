import torch
import torch.nn as nn
# from utils import Channels, SNR_to_noise
# from torch.utils.data import DataLoader
import os
import time
# import math
from tqdm import tqdm
import argparse
from dataset import AEinput
from torch.utils.data import DataLoader


parser  = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=2, type = int)
parser.add_argument('--checkpoint_path', default='checks/ae', type = str)
parser.add_argument('--data', default="/kaggle/input/datasc/aedata.csv", type=str)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mse(x, xhat):
    loss = nn.MSELoss()
    return loss(x, xhat)

def gen_rand_vec(size, interval): #size tuple (b,l,d), interval tuple [-2,2]
    r1 = interval[0]
    r2 = interval[1]
    b = size[0]
    l = size[1]
    d = size[2]
    ip = (r1 - r2) * torch.rand(b, l, d, requires_grad=True) + r2
    # op = torch.tensor(ip, requires_grad=False)
    op = ip.clone().detach()
    # return torch.FloatTensor(b, l).uniform_(r1, r2)

    return ip, op
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(128, 256),  #128 is the op dim of Sem embedding
                                             #nn.ELU(inplace=True),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(256, 128), ###########
                                             nn.ReLU(inplace=True),##########
                                             nn.Linear(128, 16)) # gives op 128(len)*50*128(op dim)
        self.decoder = nn.Sequential(nn.Linear(16, 128),  #16 is the op dim of encoder
                                                #nn.ELU(inplace=True),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(128, 256), ###########
                                                nn.ReLU(inplace=True),##########
                                                nn.Linear(256, 128))
    def forward(self, ip):
        enc = self.encoder(ip)
        dec = self.decoder(enc)
        return enc, dec

def ae_train_step(autoencoder, ip, op, opt):
    autoencoder.train()
    opt.zero_grad()
    # print("Encoding")
    # print(ip.is_cuda, op.is_cuda)
    # ip = ip.to(device)
    # op = op.to(device)
    enc, dec = autoencoder(ip) #y
    # Auto encoded input should pass thru Channel
    #enc->yhat->dec
    # if channel == 'AWGN':
    #     yhat = channels.AWGN(enc, n_var)
    # elif channel == 'Rayleigh':
    #     yhat = channels.Rayleigh(enc, n_var)
    # elif channel == 'Rician':
    #     yhat = channels.Rician(enc, n_var)
    # else:
    #     raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
    
    # dec = autoencoder.decoder(enc) #yhat is received vec
    # print("Decoded")
    loss = mse(op, dec)
    # print(loss)
    
    loss.backward(retain_graph=True)#retain_graph=True
    # print("Backpropped")
    opt.step()

    return loss.item(), dec
    
def ae_val_step(net,ip, op):
    
    # enc = net.encoder(ip)
    # dec = net.decoder(enc) #yhat is received vec
    enc, dec = autoencoder(ip)
    loss = mse(op, dec)
    return loss.item(), ip, dec

def train_ae(epoch, autoencoder, args, opt, ip, op): #x, y, optimiser; ip==op
        """Load data"""
        dset = AEinput(ip)
        # print(dset.data[:10])
        # train_dataset = dset.train
        train_dataset = dset
        train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
                                    pin_memory=True)
        print("In epoch{}".format(epoch))
        print("Data Loaded")
        pbar = tqdm(train_iterator)    
        for sents in pbar: #batch
            sents = sents.to(device)
            # print(device)
            a = time.time()
            loss, dec = ae_train_step(autoencoder, sents, sents, opt)

            # print("trained")
            # print(dec.shape)
            # print(sents.shape)
            # _, predicted = torch.max(dec.data, 1)
            # print(predicted.shape)
            # print(dec)            
            # correct = (predicted == sents).sum().item()
            acc = torch.sum(abs(dec - sents) < 0.1)/len(dset)
            
            
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; Acc: {:.5f}%'.format(
                    epoch + 1, loss, acc
                )
            )
            # print(time.time() - a)
    

def validate_ae(epoch, net, ip, op):
    
    # dset = AEinput(args.data)
    # test_dataset = dset.train
    test_dataset = AEinput(ip)
    test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss, ip, op = ae_val_step(net, sents, sents)

            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    return total/len(test_iterator), ip, op

def load_model(model, checkpoint_path):
    model_paths = []
    for fn in os.listdir(checkpoint_path):
        if not fn.endswith('.pth'): continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])  # read the idx of image
        model_paths.append((os.path.join(checkpoint_path, fn), idx))

    model_paths.sort(key=lambda x: x[1])  # sort the image by the idx

    model_path, _ = model_paths[-1]
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    print('Model loaded!')

    return model

if __name__ == '__main__':
    a = 12800 #batch size?
    b = 26 #len of i/p vector ie., len of 1 op vector from Sem encoder 
    c = 128
    ip, op = gen_rand_vec((a, b, c), (0,1)) # op==ip ie., the data m is the target vec
    
    args, unknown = parser.parse_known_args()
        
    # ip.to(device)
    # op.to(device)
    
    """Define channel"""
    # channel = "Rayleigh"
    # channels = Channels()
    # noise_std = [SNR_to_noise(12)]
    # n_var = noise_std[0]

    
    """Define optimizer"""
    
    autoencoder = AE().to(device)
    # print("output shape:", autoencoder(ip[0])[0].shape)
    
    # optimizer = torch.optim.Adam(autoencoder.parameters(),
    #                              lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    # optimizer = torch.optim.Adam(autoencoder.parameters(),
    #                          lr = 1e-1,
    #                          weight_decay = 1e-8)


    # """Generate val set"""
    aval = 1000
    ipval, opval = gen_rand_vec((aval, b, c), (0,1))
    
    for epoch in (range(args.epochs)):
        record_acc = 10
        train_ae( epoch, autoencoder, args, optimizer, ip, op)
        
        
        val_loss, valip, valop = validate_ae(epoch, autoencoder, ipval, opval)

        if val_loss < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            with open(args.checkpoint_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(autoencoder.state_dict(), f)
            record_acc = val_loss #progressively records decreasing losses
    record_loss = []

    #accuracy reflects ip range 0,1
    """Test"""

    """Generate Test set"""
    atest = 500
    iptest, optest = gen_rand_vec((atest, b, c), (0,1))
    args.checkpoint_path = args.ae_cpath
    model = AE().to(device)
    model = load_model(model, args.checkpoint_path)
    loss, valip, valop = validate_ae(1, model, iptest, optest)
    print(loss)
    # print(valip)
    # print(valop)
    # print(model)