import tensorflow as tf
import numpy as np
import keras
import torch


x1 = torch.randn(1, 3, 7, 7)
patches = torch.nn.functional.unfold(x1, kernel_size = 5, stride = 1, padding = 0)
print(patches.shape)


