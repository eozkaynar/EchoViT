import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from echovit.dataset.vivitecho import ViViTecho

ds = ViViTecho(root="/home/eda/Desktop/EE543-Term-Project/Video-Vision-Transformer/data", split="train",
               length=32, period=1, clips=1)
vid, ef = ds[0]
print(vid.shape)  # (3, 32, H, W)
print(ef)