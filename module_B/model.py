import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class DCENet(nn.Module):
    """Zero-DCE Network"""

    def __init__(self, num_iterations=8):
        super(DCENet, self).__init__()
        self.num_iterations = num_iterations

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)

        # Decoder
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(32, 3 * num_iterations, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))

        # Decoder
        x5 = self.relu(self.conv5(x4))
        x6 = self.relu(self.conv6(x5))
        A = self.tanh(self.conv7(x6))

        # Apply curve iteratively
        enhanced = x
        for i in range(self.num_iterations):
            curve_params = A[:, i*3:(i+1)*3, :, :]
            enhanced = enhanced + curve_params * enhanced * (1 - enhanced)

        return enhanced, A
    
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# checkpoint_path = 'C:/Users/admin/ProjectLab/module_B/real_synthetic_charbonnier_perceptual_color_exposure_best.pth'
# checkpoint = torch.load(checkpoint_path)
# model = DCENet(num_iterations=8).to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# # Load & preprocess image
# img_path = './test.jpg'   # <-- ảnh của bạn
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img = img.astype(np.float32) / 255.0          # normalize [0,1]
# img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
# img_tensor = img_tensor.to(device)

# # Inference
# with torch.no_grad():
#     enhanced, _ = model(img_tensor)
#     enhanced = torch.clamp(enhanced, 0, 1)

# # Visualize
# enhanced_img = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()

