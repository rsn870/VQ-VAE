import numpy as np
import pandas as pd
import glob
import os
import torch.nn as nn
import torch
import torch.nn.functional as F

#Implemetations of https://arxiv.org/pdf/1711.00937.pdf and http://papers.nips.cc/paper/9625-generating-diverse-high-fidelity-images-with-vq-vae-2.pdf

#Code adapted from https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py and https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb


class Vector_quantize(nn.Module) :
	def __init__(self,embedding_dim , num_embeddings , commitment_cost) :
		self.dim =embedding_dim             #Dimension of the quantized space
		self.n_embed = num_embeddings       #Number of vectors in the codebook(quantized space)
		self.commitment = commitment_cost        #This is the beta parameter in the papers used to ensure tractability

		self.embedding = nn.Embedding(self.n_embed, self.dim)  #A simple lookup table that stores embeddings of a fixed dictionary and size.



		self.embedding.weight.data.uniform_(-1/self.n_embed, 1/self.n_embed) #the learnable weights of the module of shape (num_embeddings, embedding_dim)

	def forward(self,inputs) :
		#inputs should be in BHWC format 
		#if inputs are in BCHW format use this 
		#inputs = inputs.permute(0, 2, 3, 1).contiguous()

		flatten_input = inputs.view(-1, self.dim)  #Reshaping / flattening the input by using view

		distances = (torch.sum(flatten_input**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(flatten_input,self.embedding.weight.t()))
		# expanding difference in terms of L2 norm (As per paper difference between quantized vectors needs to be considered (like Nearest neighbors))

		#Taking care of the embeddings

		#Nearest embedding 

		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)



		encodings = torch.zeros(encoding_indices.shape[0] , self.n_embed , device = inputs.device())

		#Using scatter_ to set values at indices


        encodings.scatter_(1, encoding_indices, 1)


        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())  #Can use nn.CrossEntropy as below

        #e_latent_loss = F.cross_entropy(quantized.detach(),inputs)
        #q_latent_loss = F.cross_entropy(quantized , inputs.detach()) 

        #detach() detaches the output from the computational graph. So no gradient will be backproped along this variable. Reqd acc to eqn in paper

        #Implementing the codebook loss from the papers
        
        loss = q_latent_loss + self.commitment * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))) #perplexity metric is same as that in NLP (uncertainty using Shannon Entropy)

        #quantized is in BHWC format if inputs are in BCHW format use this 

        #quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return loss , quantized , perplexity , encodings

        #An EMA version was also implemented in most papers but I have not used it since I'm not familiar with EMA


		


#Creating a Residual layer 

class Residual(nn.Module) :
	def __init__(self,in_channels , num_hiddens , num_residual_hiddens) :
		super(Residual, self).__init__()

		self.block = nn.Sequential(nn.ReLU(True) , 
			nn.Conv2d(in_channels = in_channels , out_channels = num_residual_hiddens , kernel_size =3, stride = 1 , padding =1 , bias =False ),
			nn.ReLU(True) ,
			nn.Conv2d(in_channels = num_residual_hiddens , out_channels = num_hiddens , kernel_size =3, stride = 1 , padding =1 , bias =False ) )

   def forward(self,x) :

		return x + self.block(x)


# Stacking Residual Layers

class Stack_Residual(nn.Module) :
	def __init__(self,in_channels,num_hiddens , num_residual_layers , num_residual_hiddens) :
		super(Stack_Residual , self).__init__()

		self.num_residual_layers = num_residual_layers

		self.layers = nn.ModuleList([Residual(in_channels , num_hiddens , num_residual_hiddens) for _ in range(self.num_residual_layers)])

		#Stacking multiple blocks 

	def forward(self ,x) :
		for i in range(self.num_residual_layers) :
			x = self.layers[i](x)

		return F.ReLU(x)  #Using functional relu as seperate layer is not needed 



class Generator(nn.Module) :
	def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens) :
		super(Generator , self).__init__()

		self.conv1 = nn.Conv2d(in_channels=in_channels,
					out_channels=num_hiddens//2,
					kernel_size=4,
					stride=2, padding=1)

		self.conv2 = nn.Conv2d(in_channels=num_hiddens//2,
					out_channels=num_hiddens,
					kernel_size=4,
					stride=2, padding=1)

		self.conv3 = nn.Conv2d(in_channels=num_hiddens,
					out_channels=num_hiddens,
					kernel_size=3,
					stride=1, padding=1)

		self.residual_stack =  Stack_Residual(in_channels=num_hiddens,
					num_hiddens=num_hiddens,
					num_residual_layers=num_residual_layers,
					num_residual_hiddens=num_residual_hiddens)

		self.conv4 = nn.Conv2d(in_channels=in_channels,
					out_channels=num_hiddens,
					kernel_size=3, 
					stride=1, padding=1)

		self.conv_trans1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
					out_channels=num_hiddens//2,
					kernel_size=4, 
					stride=2, padding=1)

		self.conv_trans2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
					out_channels=3,
					kernel_size=4, 
					stride=2, padding=1)

	def encoder(self , x) :
		#Code for the encoder 

		y = self.conv1(x)

		y = F.ReLU(y)

		y = self.conv2(y)

		y = F.ReLU(y)

		y = self.conv3(y)

		return self.residual_stack(y)

	def decoder(self , x) :

		#Code for the decoder

		y = self.conv4(x)

		y = self.residual_stack(y)

		y = self.conv_trans1(y)

		y = F.ReLU(y)

		y = self.conv_trans2(y)

		return y



class Model(nn.Module) :
	def __init__(self , in_channels , num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost) :

		super(Model , self).__init__()

		self.enc_gen = Generator(in_channels, num_hiddens,num_residual_layers, num_residual_hiddens)

		self.prevq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim,kernel_size=1 ,stride=1)

		self.dec_gen = Generator(embedding_dim,num_hiddens, num_residual_layers,num_residual_hiddens)

		self.vqvae = Vector_quantize(embedding_dim , num_embeddings , commitment_cost)


	def forward(self , x) :

	 z = self.enc_gen.encoder(x)

	 z = self.prevq_conv(z)     

	 loss, quantized, perplexity, _ = self.vqvae(z)     #Passed through vqvae

	 x_recon = self.dec_gen.decoder(quantized) #Reconstructed x

	 return loss , x_recon , perplexity















		
		






















		



		







