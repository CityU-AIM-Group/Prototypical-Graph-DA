import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminator_O(nn.Module):

	def __init__(self, in_channel, ndf = 64):
		super(FCDiscriminator_O, self).__init__()

		self.conv1 = nn.Conv2d(in_channel, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x

class FCDiscriminator_F(nn.Module):

	def __init__(self, in_channel):
		super(FCDiscriminator_F, self).__init__()

		self.conv1 = nn.Conv2d(in_channel, 1024, kernel_size=1, stride=1, padding=1)
		self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
		self.classifier = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)

		return x
