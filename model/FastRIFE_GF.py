import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from model.IFNet import *
import torch.nn.functional as F
from model.loss import *
import cv2
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )
# 1. The first convolutional layer is a 3x3 convolutional layer with 16 output channels.
# 2. The second convolutional layer is a 1x1 convolutional layer with 32 output channels.
# 3. The third convolutional layer is a 3x3 convolutional layer with 32 output channels.
# 4. The fourth convolutional layer is a 1x1 convolutional layer with 64 output channels.
# 5. The fifth convolutional layer is a 3x3 convolutional layer with 64 output channels.
# 6. The sixth convolutional layer is a 1x1 convolutional layer with 128 output channels.

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
    )
# 1. The first layer is a ConvTranspose2d layer with in_planes and out_planes.
# 2. The second layer is a PReLU layer with out_planes.

def conv_woact(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
    )
# 1. The constructor is creating a sequential model that contains a convolutional layer with the input parameters.
# 2. The forward function is returning the convolutional layer's output.

class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(ResBlock, self).__init__()
        if in_planes == out_planes and stride == 1:
            self.conv0 = nn.Identity()
        else:
            self.conv0 = nn.Conv2d(in_planes, out_planes,
                                   3, stride, 1, bias=False)
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv_woact(out_planes, out_planes, 3, 1, 1)
        self.relu1 = nn.PReLU(1)
        self.relu2 = nn.PReLU(out_planes)
        self.fc1 = nn.Conv2d(out_planes, 16, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(16, out_planes, kernel_size=1, bias=False)
# 1. If the number of input and output channels are the same and the stride is 1,
#    then the first convolution is identity.
# 2. Otherwise, the first convolution is a normal convolution with the given
#    parameters.
# 3. The second convolution is a normal convolution with the given parameters.
# 4. The third convolution is a 1x1 convolution with 16 channels.
# 5. The fourth convolution is a 1x1 convolution with the number of channels.

    def forward(self, x):
        y = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        w = x.mean(3, True).mean(2, True)
        w = self.relu1(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        x = self.relu2(x * w + y)
        return x
# 1. The first convolutional layer is a 3x3 convolutional layer with 64 filters and stride 1.
# 2. The second convolutional layer is a 3x3 convolutional layer with 64 filters and stride 1.
# 3. The third convolutional layer is a 1x1 convolutional layer with 64 filters and stride 1.
# 4. The fourth convolutional layer is a 1x1 convolutional layer with 64 filters and stride 1.

c = 16

class ContextNet(nn.Module):
    def __init__(self):
        super(ContextNet, self).__init__()
        self.conv1 = ResBlock(3, c)
        self.conv2 = ResBlock(c, 2*c)
        self.conv3 = ResBlock(2*c, 4*c)
        self.conv4 = ResBlock(4*c, 8*c)
# 1. The constructor is initializing the layers of the network.
# 2. The forward function is passing the input through each layer and returning the output.

    def forward(self, x, flow):
        x = self.conv1(x)
        f1 = warp(x, flow)
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]
# 1. The first convolutional layer is a 3x3 convolutional layer with 64 filters.
# 2. The second convolutional layer is a 3x3 convolutional layer with 64 filters.
# 3. The third convolutional layer is a 3x3 convolutional layer with 64 filters.
# 4. And so on

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.down0 = ResBlock(8, 2*c)
        self.down1 = ResBlock(4*c, 4*c)
        self.down2 = ResBlock(8*c, 8*c)
        self.down3 = ResBlock(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 4, 3, 1, 1)
# 1. The first layer is a regular convolutional layer with 8 input channels and 2 output channels.
# 2. The second layer is a regular convolutional layer with 4 input channels and 4 output channels.
# 3. The third layer is a regular convolutional layer with 8 input channels and 8 output channels.
# 4. The fourth layer is a regular convolutional layer with 16 input channels and 16 output channels.
# 5. The fifth layer is a deconvolutional layer with 16 input channels and 8 output channels.
# 6. The sixth layer is a deconvolutional layer with 8 input channels and 4 output channels.
# 7. The seventh layer is a deconvolutional layer with 4 input channels and 2 output channels.
# 8. The eighth layer is a deconvolutional layer with 2 input channels and 1 output channel.
# 9. The last layer is a regular convolutional layer with 2 input channels and 4 output channels.

    def forward(self, img0, img1, flow, c0, c1, flow_gt):
        warped_img0 = warp(img0, flow)
        warped_img1 = warp(img1, -flow)
        if flow_gt == None:
            warped_img0_gt, warped_img1_gt = None, None
        else:
            warped_img0_gt = warp(img0, flow_gt[:, :2])
            warped_img1_gt = warp(img1, flow_gt[:, 2:4])
        s0 = self.down0(torch.cat((warped_img0, warped_img1, flow), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return x, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt
# 1. The first two lines are simply initializing the class.
# 2. The next four lines define the five down-sampling layers.
# 3. The next four lines define the five up-sampling layers.
# 4. The next two lines define the two convolutional layers.
# 5. The next two lines define the two transposed convolutional layers.
# 6. The next two lines define the two fully-connected layers.

class Model:
    def __init__(self, local_rank=0, training=False):
        self.contextnet = ContextNet()
        self.fusionnet = FusionNet()
        self.device()
        self.optimG = AdamW(itertools.chain(
            self.contextnet.parameters(),
            self.fusionnet.parameters()), lr=1e-6, weight_decay=1e-5)
        self.schedulerG = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimG, patience=5, factor=0.2, verbose=True)
        self.epe = EPE()
        self.ter = Ternary()
        self.sobel = SOBEL()
        if local_rank != -1:
            pass
# 1. Initialize the ContextNet and FusionNet modules.
# 2. Initialize the optimizer for the ContextNet and FusionNet modules.
# 3. Initialize the scheduler for the optimizer.
# 4. Initialize the EPE, Ternary, and SOBEL modules.
# 5. Initialize the local rank.

    def train(self):
        self.contextnet.train()
        self.fusionnet.train()

    def eval(self):
        self.contextnet.eval()
        self.fusionnet.eval()

    def device(self):
        self.contextnet.to(device)
        self.fusionnet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == 0:
                param2 = {}
                for k, v in param.items():
                    if 'module.' in k:
                        k = k.replace("module.", "")
                    param2[k] = v
                return param2
            else:
                return param
        if rank <= 0:
            self.contextnet.load_state_dict(
                convert(torch.load('{}/contextnet.pkl'.format(path), map_location=device)))
            self.fusionnet.load_state_dict(
                convert(torch.load('{}/unet.pkl'.format(path), map_location=device)))
# 1. The class has two functions, load_model and save_model.
# 2. The load_model function takes in the path of the checkpoint and the rank of the current process.
# 3. If the rank is 0, it loads the model parameters from the checkpoint.
# 4. If the rank is not 0, it loads the model parameters from the checkpoint.
# 5. The save_model function takes in the path of the checkpoint and the rank of the current process.
# 6. If the rank is 0, it saves the model parameters to the checkpoint.
# 7. If the rank is not 0, it does nothing.

    def save_model(self, path, rank):
        if rank == 0:
            torch.save(self.contextnet.state_dict(), '{}/contextnet.pkl'.format(path))
            torch.save(self.fusionnet.state_dict(), '{}/unet.pkl'.format(path))

    def predict(self, imgs, flow, training=True, flow_gt=None):
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        c0 = self.contextnet(img0, flow)
        c1 = self.contextnet(img1, -flow)
        flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear",
                             align_corners=False) * 2.0
        refine_output, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt = self.fusionnet(
            img0, img1, flow, c0, c1, flow_gt)
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_img0 * mask + warped_img1 * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)
        pred = pred.type(torch.HalfTensor)
        mask = mask.type(torch.HalfTensor)
        merged_img = merged_img.type(torch.HalfTensor)
        warped_img0 = warped_img0.type(torch.HalfTensor)
        warped_img1 = warped_img1.type(torch.HalfTensor)
        if warped_img0_gt is not None:
            warped_img0_gt = warped_img0_gt.type(torch.HalfTensor)
            warped_img1_gt = warped_img1_gt.type(torch.HalfTensor)
        if training:
            return pred, mask, merged_img, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt
        else:
            return pred
# 1. The contextnet is a pretrained model that extracts features from an image.
# 2. The fusionnet is a pretrained model that takes in two images and their features and outputs a refined output.
# 3. The refined output is a refined output that is a combination of the warped images and residuals.
# 4. The residuals are the difference between the warped images and the actual images.
# 5. The output of the fusionnet is a tuple of refined output, warped image 0, warped image 1, warped image 0 ground truth, warped image 1 ground truth.
# 6. The refined output is a tuple of refined output, mask, merged image, warped image 0, warped image 1, warped image 0 ground truth, warped image 1 ground truth.
# 7. The refined output is the output of the fusionnet.
# 8. The mask is the mask that is used to merge the images.
# 9. The merged image is the image that is generated by merging the warped images.
# 10. The warped image 0 is the image that is generated by warping the image 0 with the flow.
# 11. The warped image 1 is the image that is generated by warping the image 1 with the flow.

    def inference(self, img0, img1):
        imgs = torch.cat((img0, img1), 1)
        flow = self.calculate_flow(imgs)
        flow = flow.to(device)
        return self.predict(imgs, flow, training=False)

    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()

        flow = Variable(self.calculate_flow(imgs).to(device))
        pred, mask, merged_img, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt = self.predict(
            imgs, flow, flow_gt=flow_gt)
        gt = gt.type(torch.HalfTensor)
        loss_ter = self.ter(pred, gt).mean()
        if training:
            with torch.no_grad():
                loss_mask = torch.abs(merged_img - gt).sum(1, True).float().detach()
                loss_mask = F.interpolate(loss_mask, scale_factor=0.5, mode="bilinear",
                                          align_corners=False).detach()
        else:
            loss_mask = 1
        loss_l1 = (((pred.type(torch.FloatTensor) - gt.type(torch.FloatTensor)) ** 2 + 1e-6) ** 0.5).mean()

        loss_ter = loss_ter.sum()
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_ter
            loss_G.backward()
            self.optimG.step()
        loss_flow = 0
        return pred, merged_img, flow, loss_l1, loss_flow, loss_ter, loss_mask
# 1. The update function takes in the input images and ground truth, and calculates the flow using the calculate_flow function.
# 2. The predict function takes in the input images and the flow, and returns the predicted output.
# 3. The calculate_flow function calculates the flow using the two images and the flow_gt.
# 4. The ter function calculates the TER loss between the predicted and ground truth images.
# 5. The train function sets the model to training mode, and the eval function sets the model to evaluation mode.
# 6. The loss_mask is set to 1 if we are in evaluation mode, and is set to the absolute difference between the merged image and the ground truth image if we are in training mode.
# 7. The loss_l1 is the L1 loss between the predicted and ground truth images.
# 8. The loss_ter is the TER loss between the predicted and ground truth images.
# 9. The loss_G is the sum of the loss_l1 and loss_ter.
# 10. The optimG.zero_grad() function zeroes out the gradient of the parameters of the generator.
# 11. The loss_G.backward() function backpropagates the loss through the generator.
# 12. The optimG.step() function updates the parameters of the generator.

    def calculate_flow(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear",
                          align_corners=False)
        img0 = x[:, :3].cpu().numpy()
        img1 = x[:, 3:].cpu().numpy()

        num_samples, _, x, y = img0.shape
        flow_batch = np.empty((0, 2, x, y))
        flow_time = []
        for i in range(num_samples):
            img0_single = img0[i, :, :, :].reshape(x, y, 3)
            img1_single = img1[i, :, :, :].reshape(x, y, 3)
            img0_single = cv2.cvtColor(img0_single, cv2.COLOR_BGR2GRAY)
            img1_single = cv2.cvtColor(img1_single, cv2.COLOR_BGR2GRAY)

            start2 = time.time()
            flow_single = cv2.calcOpticalFlowFarneback(img0_single, img1_single, None, pyr_scale=0.2, levels=3,
                                                       winsize=15, iterations=1, poly_n=1, poly_sigma=1.2, flags=0)
            end2 = time.time()
            flow_time.append((end2 - start2) * 1000)
            flow_single = flow_single.reshape(1, 2, x, y)
            flow_batch = np.append(flow_batch, flow_single, axis=0)
        return torch.tensor(flow_batch, dtype=torch.float)
# 1. The first line is simply importing the necessary libraries.
# 2. The second line is defining the class Model.
# 3. The third line is defining the function calculate_flow.
# 4. The function calculate_flow() takes in the input images and calculates the optical flow between the images.
# 5. The optical flow is calculated using the OpenCV function calcOpticalFlowFarneback().
# 6. The optical flow is calculated in batches.
# 7. The optical flow is returned as a tensor.

if __name__ == '__main__':
    img0 = torch.zeros(3, 3, 256, 256).float().to(device)
    img1 = torch.tensor(np.random.normal(
        0, 1, (3, 3, 256, 256))).float().to(device)
    imgs = torch.cat((img0, img1), 1)
    model = Model()
    model.eval()
    print(model.inference(imgs).shape)
