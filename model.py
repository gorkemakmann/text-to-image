import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import  autograd
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import os
import io
import h5py
import numpy as np
import pdb
from PIL import Image
import matplotlib.pyplot as plt


from functools import partial



class Text2ImageDataset(Dataset):

    def __init__(self, datasetFile, transform=None, split=0):
        self.datasetFile = datasetFile
        self.transform = transform
        self.dataset = None
        self.dataset_keys = None
        self.split = 'train' if split == 0 else 'valid' if split == 1 else 'test'
        self.h5py2int = self.process_int_lambda

    def __len__(self):
        f = h5py.File(self.datasetFile, 'r')
        self.dataset_keys = [str(k) for k in f[self.split].keys()]
        length = len(f[self.split])
        f.close()

        return length
    
    def process_int_lambda(self,x):
        return partial(int, np.array(x))
    
    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.datasetFile, mode='r')
            self.dataset_keys = [str(k) for k in self.dataset[self.split].keys()]

        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]

        # pdb.set_trace()

        right_image = bytes(np.array(example['img']))
        right_embed = np.array(example['embeddings'], dtype=float)
        wrong_image = bytes(np.array(self.find_wrong_image(example['class'])))
        inter_embed = np.array(self.find_inter_embed())

        right_image = Image.open(io.BytesIO(right_image)).resize((64, 64))
        wrong_image = Image.open(io.BytesIO(wrong_image)).resize((64, 64))

        right_image = self.validate_image(right_image)
        wrong_image = self.validate_image(wrong_image)

        txt = np.array(example['txt']).astype(str)

        sample = {
                'right_images': torch.FloatTensor(right_image),
                'right_embed': torch.FloatTensor(right_embed),
                'wrong_images': torch.FloatTensor(wrong_image),
                'inter_embed': torch.FloatTensor(inter_embed),
                'txt': str(txt)
                 }

        sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5)
        sample['wrong_images'] =sample['wrong_images'].sub_(127.5).div_(127.5)

        return sample

    def find_wrong_image(self, category):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        _category = example['class']

        if _category != category:
            return example['img']

        return self.find_wrong_image(category)

    def find_inter_embed(self):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        return example['embeddings']


    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)
    

class Concat_embed(nn.Module):

    def __init__(self, embed_dim, projected_embed_dim):
        super(Concat_embed, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=projected_embed_dim),
            nn.BatchNorm1d(num_features=projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, inp, embed):
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(4, 4, 1, 1).permute(2,  3, 0, 1)
        hidden_concat = torch.cat([inp, replicated_embed], 1)

        return hidden_concat


class Utils(object):

    @staticmethod
    def smooth_label(tensor, offset):
        return tensor + offset


    @staticmethod
    def save_checkpoint(netD, netG, dir_path, subdir_path, epoch):
        path =  os.path.join(dir_path, subdir_path)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(netD.state_dict(), '{0}/discriminator_{1}.pth'.format(path, epoch))
        torch.save(netG.state_dict(), '{0}/generator_{1}.pth'.format(path, epoch))

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10
noise_dim = 100
batch_size = 64
num_workers = 2
lr = 0.0002
beta1 = 0.5
# DITER = 5
l1_coef = 50
l2_coef = 100
print(device)

dataset_path = "C:/Users/gorke/OneDrive/Masaüstü/WebApp/dataset/flowers.hdf5"
train_dataset = Text2ImageDataset(dataset_path,split=0)
test_dataset = Text2ImageDataset(dataset_path,split = 3)
vaild_dataset = Text2ImageDataset(dataset_path,split = 1)
train_dataset = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=num_workers )
test_dataset = DataLoader(test_dataset,batch_size=batch_size,shuffle=True, num_workers=num_workers )
vaild_dataset = DataLoader(vaild_dataset,batch_size=batch_size,shuffle=True, num_workers=num_workers)


            
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.channels = 3
        self.noise_d = 100
        self.text_dim = 1024
        self.reduced_text_dim = 128
        self.num_features = 64
        
        self.text = nn.Sequential(
            nn.Linear(in_features=self.text_dim,out_features=self.reduced_text_dim),
            nn.BatchNorm1d(num_features=self.reduced_text_dim),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.noise_d + self.reduced_text_dim,out_channels= self.num_features * 8, kernel_size=4, stride= 1, padding = 0,bias=False),
            nn.BatchNorm2d(self.num_features * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=self.num_features * 8 ,out_channels= self.num_features * 4, kernel_size=4, stride= 2, padding = 1,bias=False),
            nn.BatchNorm2d(self.num_features * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels= self.num_features * 4 ,out_channels= self.num_features * 2, kernel_size=4, stride= 2, padding = 1,bias=False),
            nn.BatchNorm2d(self.num_features * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels= self.num_features * 2 ,out_channels= self.num_features, kernel_size=4, stride= 2, padding = 1,bias=False),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=self.num_features, out_channels= self.channels,kernel_size=4, stride= 2 , padding= 1, bias= False),
            nn.Tanh()
        )
        
    def forward(self,text_vector,noise):
        text = self.text(text_vector).unsqueeze(2).unsqueeze(3)
        combined_text_noise = torch.cat([text,noise],1)
        return self.generator(combined_text_noise)
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.image_size = 256
        self.channels = 3
        self.text_dim = 1024
        self.reduced_text = 128
        self.num_features = 64
        
        self.image_features = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels= self.num_features, kernel_size=4, stride=2 , padding=1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(in_channels= self.num_features, out_channels= self.num_features * 2, kernel_size=4 , stride=2, padding= 1, bias=False),
            nn.BatchNorm2d(self.num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels= self.num_features * 2, out_channels= self.num_features * 4, kernel_size=4 , stride=2, padding= 1, bias=False),
            nn.BatchNorm2d(self.num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels= self.num_features * 4, out_channels= self.num_features * 8, kernel_size=4 , stride=2, padding= 1, bias=False),
            nn.BatchNorm2d(self.num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.concat = Concat_embed(self.text_dim, self.reduced_text)
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=self.num_features * 8 + self.reduced_text, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, image, text):
        image_features = self.image_features(image)
        
        x = self.concat(image_features,text)
        x = self.discriminator(x)
        
        return x.view(-1,1).squeeze(1), image_features

generator = nn.DataParallel( Generator().to(device))
generator.apply(Utils.weights_init)
generator.to(device)

discriminator = nn.DataParallel(Discriminator().to(device))
discriminator.apply(Utils.weights_init)
discriminator.to(device)

optimD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

import torch
# Load the .pth file
generator_pth = 'C:/Users/gorke/OneDrive/Masaüstü/WebApp/saved-model/generator_15.pth'

# Create a new instance of the Discriminator class
generator = nn.DataParallel(Generator().to(device))

# Set the parameters of the discriminator to the values loaded from the .pth file
generator.load_state_dict(torch.load(generator_pth))

# Load the .pth file
discriminator_pth = 'C:/Users/gorke/OneDrive/Masaüstü/WebApp/saved-model/discriminator_15.pth'

# Create a new instance of the Discriminator class
discriminator = nn.DataParallel(Discriminator().to(device))

# Set the parameters of the discriminator to the values loaded from the .pth file
discriminator.load_state_dict(torch.load(discriminator_pth))

generator.eval()
discriminator.eval()

from PIL import Image
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_text = request.form['input_text']
        return redirect(url_for('generate_images', input_text=input_text))
    return render_template('index.html')

@app.route('/generate_images/<input_text>', methods=['GET'])
def generate_images(input_text):
    for sample in test_dataset:
        right_images = sample['right_images']
        right_embed = sample['right_embed']
        txt = input_text

        right_images = Variable(right_images.float()).to(device)
        right_embed = Variable(right_embed.float()).to(device)
        noise = Variable(torch.randn(right_images.size(0), 100)).to(device)
        noise = noise.view(noise.size(0), 100, 1, 1)
        fake_images = generator(right_embed, noise)
        for image, t in zip(fake_images, txt):
            im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            # Generate a unique filename for each image using the text
            filename = f"{input_text}.png"

            # Save the image
            file_path = f"static/generated_images/{filename}"
            im.save(file_path)
            break
        break

    return render_template('result.html', image_filename=filename, input_text=input_text)

import uuid

# Generate a GUID
guid = uuid.uuid4()

# Convert the GUID to string format
guid_str = str(guid)

# Print the GUID
print(guid_str)
@app.route('/generate_test_images/', methods=['POST'])
def generate_test_images():
    for sample in test_dataset:
        right_images = sample['right_images']
        right_embed = sample['right_embed']
        txt = sample['txt']

        right_images = Variable(right_images.float()).to(device)
        right_embed = Variable(right_embed.float()).to(device)
        noise = Variable(torch.randn(right_images.size(0), 100)).to(device)
        noise = noise.view(noise.size(0), 100, 1, 1)
        fake_images = generator(right_embed, noise)
        for image, t in zip(fake_images, txt):
            im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            # Generate a unique filename for each image using the text
            filename = f"{guid_str}.png"

            # Save the image
            file_path = f"static/generated_test_images/{filename}"
            im.save(file_path)
            break
        break

    return render_template('test_result.html', image_filename=filename, t=t)

if __name__ == '__main__':
    app.run(debug=True)