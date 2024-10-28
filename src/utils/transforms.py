# from torchvision.utils import _log_api_usage_once
import numpy as np
from utils.helpers import get_bit_plane
import torch
from torchvision.utils import _log_api_usage_once

# def _log_api_usage_once(x):
#     return

import numpy as np
import torch
from PIL import Image

from PIL import Image
import numpy as np
import torch

class PILToTensorUint8:
    def __init__(self, size=(32, 32)) -> None:
        self.size = size  # 设置目标图像大小

    def __call__(self, pic):
        # 将图像调整为目标大小
        pic = pic.resize(self.size, Image.BILINEAR)  # 使用双线性插值调整大小

        # 如果不是深度为8的灰度图，转换为灰度图
        if pic.mode != 'L':  # 'L' 表示灰度图
            pic = pic.convert('L')  # 转换为灰度图

        # 将图像转换为 numpy 数组
        np_img = np.array(pic)
        
        # 检查深度是否为8位（0-255范围），如果不是，进行转换
        if np_img.dtype != np.uint8:
            np_img = (np_img * 255).astype(np.uint8)  # 确保数据在0-255范围内并转换为uint8

        # 手动扩展维度，使其符合 (C, H, W) 的格式
        np_img = np.expand_dims(np_img, axis=2)  # 扩展一个维度，使其变成 (H, W, 1)

        # 转换为 PyTorch tensor 并调整通道顺序
        return torch.Tensor(np_img).to(torch.uint8).permute((2, 0, 1)).contiguous()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"




class ToTensorUint8:
    def __init__(self) -> None:
        _log_api_usage_once(self)

    # @staticmethod
    def __call__(self, pic):     
        return torch.Tensor(pic).to(torch.uint8)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class GetSubPlane:
    def __init__(self, plane:int) -> None:
        _log_api_usage_once(self)
        self.plane = plane

    # @staticmethod
    def __call__(self, x):     
        return get_bit_plane(x, self.plane)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
