import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings

from model.FastRIFE_GF import Model
path = r'model_GF'
output_path = r'output/GF/2.png'
# 1. First, it loads the model from the path.
# 2. Then, it loads the weights from the weights path.
# 3. Then, it loads the test image from the image path.
# 4. After that, it runs the model on the test image and saves the output in the output path.

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
# 1. The first line is initializing the device to use for the model.
# 2. The second line is setting the gradient calculation to be false.
# 3. The third line is checking if a CUDA device is available.
# 4. The fourth line is setting the cuDNN auto-tuner to optimize the convolution algorithms.

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--img', dest='img', nargs=2, required=True)
parser.add_argument('--exp', default=1, type=int)
parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
args = parser.parse_args()
# 1. The constructor takes in the image paths and the interpolation ratio.
# 2. The interpolation ratio is used to calculate the interpolation factor.
# 3. The interpolation factor is used to calculate the interpolated image.
# 4. The interpolated image is displayed.

model = Model()
model.load_model(path)
model.eval()
model.device()
# 1. The class has a constructor that takes a path to a model file.
# 2. The class has a load_model method that takes a path to a model file.
# 3. The class has a eval method that sets the model to evaluation mode.
# 4. The class has a device method that sets the model to either GPU or CPU.

img0 = cv2.imread(args.img[0])
img1 = cv2.imread(args.img[1])
img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
# 1. The constructor takes in the argument parser, which is a class that stores all the arguments that are passed to the script.
# 2. The constructor then stores the arguments in a dictionary, which is a more convenient way to access the arguments.

n, c, h, w = img0.shape
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
img0 = F.pad(img0, padding)
img1 = F.pad(img1, padding)
# 1. The constructor takes in the input image and the number of channels.
# 2. The forward function takes in the input image and applies the convolutional layer.
# 3. The backward function computes the gradient of the loss with respect to the input image.

if args.ratio:
    img_list = [img0]
    img0_ratio = 0.0
    img1_ratio = 1.0
    if args.ratio <= img0_ratio + args.rthreshold / 2:
        middle = img0
    elif args.ratio >= img1_ratio - args.rthreshold / 2:
        middle = img1
    else:
        tmp_img0 = img0
        tmp_img1 = img1
        for inference_cycle in range(args.rmaxcycles):
            middle = model.inference(tmp_img0, tmp_img1)

            middle_ratio = ( img0_ratio + img1_ratio ) / 2
            if args.ratio - (args.rthreshold / 2) <= middle_ratio <= args.ratio + (args.rthreshold / 2):
                break
            if args.ratio > middle_ratio:
                tmp_img0 = middle
                img0_ratio = middle_ratio
            else:
                tmp_img1 = middle
                img1_ratio = middle_ratio
    img_list.append(middle)
    img_list.append(img1)
else:
    img_list = [img0, img1]
    for i in range(args.exp):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img1)
        img_list = tmp
# 1. The constructor takes in the model and the image paths.
# 2. The inference() function takes in the two images and returns the middle image.
# 3. The inference_cycle() function takes in the two images and returns the middle image.
# 4. The inference_exp() function takes in the two images and returns the middle image.
# 5. The inference_ratio() function takes in the two images and returns the middle image.

if not os.path.exists('output'):
    os.mkdir('output')
for i in range(len(img_list)):
    if i == 1:
        cv2.imwrite(output_path, (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
# 1. The constructor takes in the model, the image, and the number of classes.
# 2. The forward function takes in the data and applies the model to it.
# 3. The output function takes in the class scores and returns the class with the highest probability.
