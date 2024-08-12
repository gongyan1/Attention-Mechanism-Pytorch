# Attention-Mechanism-Pytorch
This repository contains an implementation of many attention mechanism models.

# Change Log
- [x] Published Initial Attention Models, 2024-8-12.

# Quick Start
## pip
```
pip install AttentionMechanism
```

## Clone Repository
```
git clone https://github.com/gongyan1/Attention-Mechanism-Pytorch
```

## Demo
```
python demo.py
```


# 目录

- [Attention Series](#attention-series)
    - [1. External Attention Usage](#1-external-attention-usage)

    - [2. Self Attention Usage](#2-self-attention-usage)

    - [3. Simplified Self Attention Usage](#3-simplified-self-attention-usage)

    - [4. Squeeze-and-Excitation Attention Usage](#4-squeeze-and-excitation-attention-usage)

    - [5. SK Attention Usage](#5-sk-attention-usage)

    - [6. CBAM Attention Usage](#6-cbam-attention-usage)

    - [7. BAM Attention Usage](#7-bam-attention-usage)
    
    - [8. ECA Attention Usage](#8-eca-attention-usage)

    - [9. DANet Attention Usage](#9-danet-attention-usage)

    - [10. Pyramid Split Attention (PSA) Usage](#10-Pyramid-Split-Attention-Usage)

    - [11. Efficient Multi-Head Self-Attention(EMSA) Usage](#11-Efficient-Multi-Head-Self-Attention-Usage)

    - [12. Shuffle Attention Usage](#12-Shuffle-Attention-Usage)
    
    - [13. MUSE Attention Usage](#13-MUSE-Attention-Usage)
  
    - [14. SGE Attention Usage](#14-SGE-Attention-Usage)

    - [15. A2 Attention Usage](#15-A2-Attention-Usage)

    - [16. AFT Attention Usage](#16-AFT-Attention-Usage)

    - [17. Outlook Attention Usage](#17-Outlook-Attention-Usage)

    - [18. ViP Attention Usage](#18-ViP-Attention-Usage)

    - [19. CoAtNet Attention Usage](#19-CoAtNet-Attention-Usage)

    - [20. HaloNet Attention Usage](#20-HaloNet-Attention-Usage)

    - [21. Polarized Self-Attention Usage](#21-Polarized-Self-Attention-Usage)

    - [22. CoTAttention Usage](#22-CoTAttention-Usage)

    - [23. Residual Attention Usage](#23-Residual-Attention-Usage)
  
    - [24. S2 Attention Usage](#24-S2-Attention-Usage)

    - [25. GFNet Attention Usage](#25-GFNet-Attention-Usage)

    - [26. Triplet Attention Usage](#26-TripletAttention-Usage)

    - [27. Coordinate Attention Usage](#27-Coordinate-Attention-Usage)

    - [28. MobileViT Attention Usage](#28-MobileViT-Attention-Usage)

    - [29. ParNet Attention Usage](#29-ParNet-Attention-Usage)

    - [30. UFO Attention Usage](#30-UFO-Attention-Usage)

    - [31. ACmix Attention Usage](#31-Acmix-Attention-Usage)
  
    - [32. MobileViTv2 Attention Usage](#32-MobileViTv2-Attention-Usage)

    - [33. DAT Attention Usage](#33-DAT-Attention-Usage)

    - [34. CrossFormer Attention Usage](#34-CrossFormer-Attention-Usage)

    - [35. MOATransformer Attention Usage](#35-MOATransformer-Attention-Usage)

    - [36. CrissCrossAttention Attention Usage](#36-CrissCrossAttention-Attention-Usage)

    - [37. Axial_attention Attention Usage](#37-Axial_attention-Attention-Usage)

    - [38. Frequency Channel Attention Usage](#38-Frequency-Channel-Attention-Usage)

    - [39. Attention Augmented Convolutional Networks Usage](#39-Attention-Augmented-Convolutional-Networks-Usage)

    - [40. Global Context Attention Usage](#40-Global-Context-Attention-Usage)

    - [41. Linear Context Transform Attention Usage](#41-Linear-Context-Transform-Attention-Usage)

    - [42. Gated Channel Transformation Usage](#42-Gated-Channel-Transformation-Usage)

    - [43. Gaussian Context Attention Usage](#43-Gaussian-Context-Attention-Usage)


- [MLP Series](#mlp-series)

    - [1. RepMLP Usage](#1-RepMLP-Usage)

    - [2. MLP-Mixer Usage](#2-MLP-Mixer-Usage)

    - [3. ResMLP Usage](#3-ResMLP-Usage)

    - [4. gMLP Usage](#4-gMLP-Usage)

    - [5. sMLP Usage](#5-sMLP-Usage)

    - [6. vip-mlp Usage](#6-vip-mlp-Usage)

***

# Attention Series
### 1. External Attention Usage
#### 1.1. Paper
["Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks"](https://arxiv.org/abs/2105.02358)

#### 1.2. Overview
![](.//AttentionMechanism/model/img/External_Attention.png)

#### 1.3. Usage Code
```python
from AttentionMechanism.model.attention.ExternalAttention import ExternalAttention
import torch

input=torch.randn(50,49,512)
ea = ExternalAttention(d_model=512,S=8)
output=ea(input)
print(output.shape)
```

***


### 2. Self Attention Usage
#### 2.1. Paper
["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf)

#### 1.2. Overview
![](.//AttentionMechanism/model/img/SA.png)

#### 1.3. Usage Code
```python
from AttentionMechanism.model.attention.SelfAttention import ScaledDotProductAttention
import torch

input=torch.randn(50,49,512)
sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
output=sa(input,input,input)
print(output.shape)
```

***

### 3. Simplified Self Attention Usage
#### 3.1. Paper
[SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks (ICML 2021)](https://proceedings.mlr.press/v139/yang21o/yang21o.pdf)

#### 3.2. Overview
![](.//AttentionMechanism/model/img/SimAttention.png)

#### 3.3. Usage Code
```python
from AttentionMechanism.model.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
import torch

input=torch.randn(50,49,512)
ssa = SimplifiedScaledDotProductAttention(d_model=512, h=8)
output=ssa(input,input,input)
print(output.shape)

```

***

### 4. Squeeze-and-Excitation Attention Usage
#### 4.1. Paper
["Squeeze-and-Excitation Networks"](https://arxiv.org/abs/1709.01507)

#### 4.2. Overview
![](.//AttentionMechanism/model/img/SE.png)

#### 4.3. Usage Code
```python
from AttentionMechanism.model.attention.SEAttention import SEAttention
import torch

input=torch.randn(50,512,7,7)
se = SEAttention(channel=512,reduction=8)
output=se(input)
print(output.shape)

```

***

### 5. SK Attention Usage
#### 5.1. Paper
["Selective Kernel Networks"](https://arxiv.org/pdf/1903.06586.pdf)

#### 5.2. Overview
![](.//AttentionMechanism/model/img/SK.png)

#### 5.3. Usage Code
```python
from AttentionMechanism.model.attention.SKAttention import SKAttention
import torch

input=torch.randn(50,512,7,7)
se = SKAttention(channel=512,reduction=8)
output=se(input)
print(output.shape)

```
***

### 6. CBAM Attention Usage
#### 6.1. Paper
["CBAM: Convolutional Block Attention Module"](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)

#### 6.2. Overview
![](.//AttentionMechanism/model/img/CBAM1.png)

![](.//AttentionMechanism/model/img/CBAM2.png)

#### 6.3. Usage Code
```python
from AttentionMechanism.model.attention.CBAM import CBAMBlock
import torch

input=torch.randn(50,512,7,7)
kernel_size=input.shape[2]
cbam = CBAMBlock(channel=512,reduction=16,kernel_size=kernel_size)
output=cbam(input)
print(output.shape)

```

***

### 7. BAM Attention Usage
#### 7.1. Paper
["BAM: Bottleneck Attention Module"](https://arxiv.org/pdf/1807.06514.pdf)

#### 7.2. Overview
![](.//AttentionMechanism/model/img/BAM.png)

#### 7.3. Usage Code
```python
from AttentionMechanism.model.attention.BAM import BAMBlock
import torch

input=torch.randn(50,512,7,7)
bam = BAMBlock(channel=512,reduction=16,dia_val=2)
output=bam(input)
print(output.shape)

```

***

### 8. ECA Attention Usage
#### 8.1. Paper
["ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"](https://arxiv.org/pdf/1910.03151.pdf)

#### 8.2. Overview
![](.//AttentionMechanism/model/img/ECA.png)

#### 8.3. Usage Code
```python
from AttentionMechanism.model.attention.ECAAttention import ECAAttention
import torch

input=torch.randn(50,512,7,7)
eca = ECAAttention(kernel_size=3)
output=eca(input)
print(output.shape)

```

***

### 9. DANet Attention Usage
#### 9.1. Paper
["Dual Attention Network for Scene Segmentation"](https://arxiv.org/pdf/1809.02983.pdf)

#### 9.2. Overview
![](.//AttentionMechanism/model/img/danet.png)

#### 9.3. Usage Code
```python
from AttentionMechanism.model.attention.DANet import DAModule
import torch

input=torch.randn(50,512,7,7)
danet=DAModule(d_model=512,kernel_size=3,H=7,W=7)
print(danet(input).shape)

```

***

### 10. Pyramid Split Attention Usage

#### 10.1. Paper
["EPSANet: An Efficient Pyramid Split Attention Block on Convolutional Neural Network"](https://arxiv.org/pdf/2105.14447.pdf)

#### 10.2. Overview
![](.//AttentionMechanism/model/img/psa.png)

#### 10.3. Usage Code
```python
from AttentionMechanism.model.attention.PSA import PSA
import torch

input=torch.randn(50,512,7,7)
psa = PSA(channel=512,reduction=8)
output=psa(input)
print(output.shape)

```

***


### 11. Efficient Multi-Head Self-Attention Usage

#### 11.1. Paper
["ResT: An Efficient Transformer for Visual Recognition"](https://arxiv.org/abs/2105.13677)

#### 11.2. Overview
![](.//AttentionMechanism/model/img/EMSA.png)

#### 11.3. Usage Code
```python

from AttentionMechanism.model.attention.EMSA import EMSA
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,64,512)
emsa = EMSA(d_model=512, d_k=512, d_v=512, h=8,H=8,W=8,ratio=2,apply_transform=True)
output=emsa(input,input,input)
print(output.shape)
    
```

***


### 12. Shuffle Attention Usage

#### 12.1. Paper
["SA-NET: SHUFFLE ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS"](https://arxiv.org/pdf/2102.00240.pdf)

#### 12.2. Overview
![](.//AttentionMechanism/model/img/ShuffleAttention.jpg)

#### 12.3. Usage Code
```python

from AttentionMechanism.model.attention.ShuffleAttention import ShuffleAttention
import torch
from torch import nn
from torch.nn import functional as F


input=torch.randn(50,512,7,7)
se = ShuffleAttention(channel=512,G=8)
output=se(input)
print(output.shape)
 
```
***


### 13. MUSE Attention Usage

#### 13.1. Paper
["MUSE: Parallel Multi-Scale Attention for Sequence to Sequence Learning"](https://arxiv.org/abs/1911.09483)

#### 13.2. Overview
![](.//AttentionMechanism/model/img/MUSE.png)

#### 13.3. Usage Code
```python
from AttentionMechanism.model.attention.MUSEAttention import MUSEAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,49,512)
sa = MUSEAttention(d_model=512, d_k=512, d_v=512, h=8)
output=sa(input,input,input)
print(output.shape)

```

***


### 14. SGE Attention Usage

#### 14.1. Paper
[Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks](https://arxiv.org/pdf/1905.09646.pdf)

#### 14.2. Overview
![](.//AttentionMechanism/model/img/SGE.png)

#### 14.3. Usage Code
```python
from AttentionMechanism.model.attention.SGE import SpatialGroupEnhance
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
sge = SpatialGroupEnhance(groups=8)
output=sge(input)
print(output.shape)

```

***


### 15. A2 Attention Usage

#### 15.1. Paper
[A2-Nets: Double Attention Networks](https://arxiv.org/pdf/1810.11579.pdf)

#### 15.2. Overview
![](.//AttentionMechanism/model/img/A2.png)

#### 15.3. Usage Code
```python
from AttentionMechanism.model.attention.A2Atttention import DoubleAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
a2 = DoubleAttention(512,128,128,True)
output=a2(input)
print(output.shape)

```


### 16. AFT Attention Usage

#### 16.1. Paper
[An Attention Free Transformer](https://arxiv.org/pdf/2105.14103v1.pdf)

#### 16.2. Overview
![](.//AttentionMechanism/model/img/AFT.jpg)

#### 16.3. Usage Code
```python
from AttentionMechanism.model.attention.AFT import AFT_FULL
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,49,512)
aft_full = AFT_FULL(d_model=512, n=49)
output=aft_full(input)
print(output.shape)

```
***


### 17. Outlook Attention Usage

#### 17.1. Paper


[VOLO: Vision Outlooker for Visual Recognition"](https://arxiv.org/abs/2106.13112)


#### 17.2. Overview
![](.//AttentionMechanism/model/img/OutlookAttention.png)

#### 17.3. Usage Code
```python
from AttentionMechanism.model.attention.OutlookAttention import OutlookAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,28,28,512)
outlook = OutlookAttention(dim=512)
output=outlook(input)
print(output.shape)

```
***


### 18. ViP Attention Usage

#### 18.1. Paper

[Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition"](https://arxiv.org/abs/2106.12368)

#### 18.2. Overview
![](.//AttentionMechanism/model/img/ViP.png)

#### 18.3. Usage Code
```python

from AttentionMechanism.model.attention.ViP import WeightedPermuteMLP
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(64,8,8,512)
seg_dim=8
vip=WeightedPermuteMLP(512,seg_dim)
out=vip(input)
print(out.shape)

```
***


### 19. CoAtNet Attention Usage

#### 19.1. Paper

[CoAtNet: Marrying Convolution and Attention for All Data Sizes"](https://arxiv.org/abs/2106.04803) 

#### 19.2. Overview
None

#### 19.3. Usage Code
```python

from AttentionMechanism.model.attention.CoAtNet import CoAtNet
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,3,224,224)
mbconv=CoAtNet(in_ch=3,image_size=224)
out=mbconv(input)
print(out.shape)

```


***

### 20. HaloNet Attention Usage

#### 20.1. Paper

[Scaling Local Self-Attention for Parameter Efficient Visual Backbones"](https://arxiv.org/pdf/2103.12731.pdf) 


#### 20.2. Overview

![](.//AttentionMechanism/model/img/HaloNet.png)

#### 20.3. Usage Code
```python

from AttentionMechanism.model.attention.HaloAttention import HaloAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,512,8,8)
halo = HaloAttention(dim=512,
    block_size=2,
    halo_size=1,)
output=halo(input)
print(output.shape)

```
***


### 21. Polarized Self-Attention Usage

#### 21.1. Paper

[Polarized Self-Attention: Towards High-quality Pixel-wise Regression"](https://arxiv.org/abs/2107.00782)  

#### 21.2. Overview

![](.//AttentionMechanism/model/img/PoSA.png)

#### 21.3. Usage Code
```python

from AttentionMechanism.model.attention.PolarizedSelfAttention import ParallelPolarizedSelfAttention,SequentialPolarizedSelfAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,512,7,7)
psa = SequentialPolarizedSelfAttention(channel=512)
output=psa(input)
print(output.shape)

```
***


### 22. CoTAttention Usage

#### 22.1. Paper

[Contextual Transformer Networks for Visual Recognition---arXiv 2021.07.26](https://arxiv.org/abs/2107.12292) 

#### 22.2. Overview

![](.//AttentionMechanism/model/img/CoT.png)

#### 22.3. Usage Code
```python

from AttentionMechanism.model.attention.CoTAttention import CoTAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
cot = CoTAttention(dim=512,kernel_size=3)
output=cot(input)
print(output.shape)

```

***


### 23. Residual Attention Usage

#### 23.1. Paper

[Residual Attention: A Simple but Effective Method for Multi-Label Recognition---ICCV2021](https://arxiv.org/abs/2108.02456) 


#### 23.2. Overview

![](.//AttentionMechanism/model/img/ResAtt.png)

#### 23.3. Usage Code
```python

from AttentionMechanism.model.attention.ResidualAttention import ResidualAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
resatt = ResidualAttention(channel=512,num_class=1000,la=0.2)
output=resatt(input)
print(output.shape)

```

***


### 24. S2 Attention Usage

#### 24.1. Paper

[S²-MLPv2: Improved Spatial-Shift MLP Architecture for Vision---arXiv 2021.08.02](https://arxiv.org/abs/2108.01072) 

#### 24.2. Overview

![](.//AttentionMechanism/model/img/S2Attention.png)

#### 24.3. Usage Code
```python
from AttentionMechanism.model.attention.S2Attention import S2Attention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
s2att = S2Attention(channels=512)
output=s2att(input)
print(output.shape)

```

***


### 25. GFNet Attention Usage

#### 25.1. Paper

[Global Filter Networks for Image Classification---arXiv 2021.07.01](https://arxiv.org/abs/2107.00645) 


#### 25.2. Overview

![](.//AttentionMechanism/model/img/GFNet.jpg)

#### 25.3. Usage Code - Implemented by [Wenliang Zhao (Author)](https://scholar.google.com/citations?user=lyPWvuEAAAAJ&hl=en)

```python
from AttentionMechanism.model.attention.gfnet import GFNet
import torch
from torch import nn
from torch.nn import functional as F

x = torch.randn(1, 3, 224, 224)
gfnet = GFNet(embed_dim=384, img_size=224, patch_size=16, num_classes=1000)
out = gfnet(x)
print(out.shape)

```

***


### 26. TripletAttention Usage

#### 26.1. Paper

[Rotate to Attend: Convolutional Triplet Attention Module---CVPR 2021](https://arxiv.org/abs/2010.03045) 

#### 26.2. Overview

![](.//AttentionMechanism/model/img/triplet.png)

#### 26.3. Usage Code - Implemented by [digantamisra98](https://github.com/digantamisra98)

```python
from AttentionMechanism.model.attention.TripletAttention import TripletAttention
import torch
from torch import nn
from torch.nn import functional as F
input=torch.randn(50,512,7,7)
triplet = TripletAttention()
output=triplet(input)
print(output.shape)
```
***


### 27. Coordinate Attention Usage

#### 27.1. Paper

[Coordinate Attention for Efficient Mobile Network Design---CVPR 2021](https://arxiv.org/abs/2103.02907)


#### 27.2. Overview

![](.//AttentionMechanism/model/img/CoordAttention.png)

#### 27.3. Usage Code - Implemented by [Andrew-Qibin](https://github.com/Andrew-Qibin)

```python
from AttentionMechanism.model.attention.CoordAttention import CoordAtt
import torch
from torch import nn
from torch.nn import functional as F

inp=torch.rand([2, 96, 56, 56])
inp_dim, oup_dim = 96, 96
reduction=32

coord_attention = CoordAtt(inp_dim, oup_dim, reduction=reduction)
output=coord_attention(inp)
print(output.shape)
```

***


### 28. MobileViT Attention Usage

#### 28.1. Paper

[MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer---ArXiv 2021.10.05](https://arxiv.org/abs/2103.02907)


#### 28.2. Overview

![](.//AttentionMechanism/model/img/MobileViTAttention.png)

#### 28.3. Usage Code

```python
from AttentionMechanism.model.attention.MobileViTAttention import MobileViTAttention
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    m=MobileViTAttention()
    input=torch.randn(1,3,49,49)
    output=m(input)
    print(output.shape)  #output:(1,3,49,49)
    
```

***


### 29. ParNet Attention Usage

#### 29.1. Paper

[Non-deep Networks---ArXiv 2021.10.20](https://arxiv.org/abs/2110.07641)


#### 29.2. Overview

![](.//AttentionMechanism/model/img/ParNet.png)

#### 29.3. Usage Code

```python
from AttentionMechanism.model.attention.ParNetAttention import *
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    pna = ParNetAttention(channel=512)
    output=pna(input)
    print(output.shape) #50,512,7,7
    
```

***


### 30. UFO Attention Usage

#### 30.1. Paper

[UFO-ViT: High Performance Linear Vision Transformer without Softmax---ArXiv 2021.09.29](https://arxiv.org/abs/2110.07641)


#### 30.2. Overview

![](.//AttentionMechanism/model/img/UFO.png)

#### 30.3. Usage Code

```python
from AttentionMechanism.model.attention.UFOAttention import *
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,49,512)
    ufo = UFOAttention(d_model=512, d_k=512, d_v=512, h=8)
    output=ufo(input,input,input)
    print(output.shape) #[50, 49, 512]
    
```

***

### 31. ACmix Attention Usage

#### 31.1. Paper

[On the Integration of Self-Attention and Convolution](https://arxiv.org/pdf/2111.14556.pdf)

#### 31.2. Usage Code

```python
from AttentionMechanism.model.attention.ACmix import ACmix
import torch

if __name__ == '__main__':
    input=torch.randn(50,256,7,7)
    acmix = ACmix(in_planes=256, out_planes=256)
    output=acmix(input)
    print(output.shape)
    
```
***

### 32. MobileViTv2 Attention Usage

#### 32.1. Paper

[Separable Self-attention for Mobile Vision Transformers---ArXiv 2022.06.06](https://arxiv.org/abs/2206.02680)


#### 32.2. Overview

![](.//AttentionMechanism/model/img/MobileViTv2.png)

#### 32.3. Usage Code

```python
from AttentionMechanism.model.attention.MobileViTv2Attention import MobileViTv2Attention
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,49,512)
    sa = MobileViTv2Attention(d_model=512)
    output=sa(input)
    print(output.shape)
    
```
***

### 33. DAT Attention Usage

#### 33.1. Paper

[Vision Transformer with Deformable Attention---CVPR2022](https://arxiv.org/abs/2201.00520)

#### 33.2. Usage Code

```python
from AttentionMechanism.model.attention.DAT import DAT
import torch

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = DAT(
        img_size=224,
        patch_size=4,
        num_classes=1000,
        expansion=4,
        dim_stem=96,
        dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        stage_spec=[['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
        heads=[3, 6, 12, 24],
        window_sizes=[7, 7, 7, 7] ,
        groups=[-1, -1, 3, 6],
        use_pes=[False, False, True, True],
        dwc_pes=[False, False, False, False],
        strides=[-1, -1, 1, 1],
        sr_ratios=[-1, -1, -1, -1],
        offset_range_factor=[-1, -1, 2, 2],
        no_offs=[False, False, False, False],
        fixed_pes=[False, False, False, False],
        use_dwc_mlps=[False, False, False, False],
        use_conv_patches=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
    )
    output=model(input)
    print(output[0].shape)
    
```
***

### 34. CrossFormer Attention Usage

#### 34.1. Paper

[CROSSFORMER: A VERSATILE VISION TRANSFORMER HINGING ON CROSS-SCALE ATTENTION---ICLR 2022](https://arxiv.org/pdf/2108.00154.pdf)

#### 34.2. Usage Code

```python
from AttentionMechanism.model.attention.Crossformer import CrossFormer
import torch

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = CrossFormer(img_size=224,
        patch_size=[4, 8, 16, 32],
        in_chans= 3,
        num_classes=1000,
        embed_dim=48,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        group_size=[7, 7, 7, 7],
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        merge_size=[[2, 4], [2,4], [2, 4]]
    )
    output=model(input)
    print(output.shape)
    
```
***

### 35. MOATransformer Attention Usage

#### 35.1. Paper

[Aggregating Global Features into Local Vision Transformer](https://arxiv.org/abs/2201.12903)

#### 35.2. Usage Code

```python
from AttentionMechanism.model.attention.MOATransformer import MOATransformer
import torch

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = MOATransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6],
        num_heads=[3, 6, 12],
        window_size=14,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False
    )
    output=model(input)
    print(output.shape)
    
```
***

### 36. CrissCrossAttention Attention Usage

#### 36.1. Paper

[CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/abs/1811.11721)

#### 36.2. Usage Code

```python
from AttentionMechanism.model.attention.CrissCrossAttention import CrissCrossAttention
import torch

if __name__ == '__main__':
    input=torch.randn(3, 64, 7, 7)
    model = CrissCrossAttention(64)
    outputs = model(input)
    print(outputs.shape)
    
```
***

### 37. Axial_attention Attention Usage

#### 37.1. Paper

[Axial Attention in Multidimensional Transformers](https://arxiv.org/abs/1912.12180)

#### 37.2. Usage Code

```python
from AttentionMechanism.model.attention.Axial_attention import AxialImageTransformer
import torch

if __name__ == '__main__':
    input=torch.randn(3, 128, 7, 7)
    model = AxialImageTransformer(
        dim = 128,
        depth = 12,
        reversible = True
    )
    outputs = model(input)
    print(outputs.shape)
    
```
***

### 38. Frequency Channel Attention Usage

#### 38.1. Paper

[FcaNet: Frequency Channel Attention Networks (ICCV 2021)](https://arxiv.org/abs/2012.11879)

#### 38.2. Overview

![](.//AttentionMechanism/model/img/FCANet.png)

#### 38.3. Usage Code

```python
from AttentionMechanism.model.attention.FCA import MultiSpectralAttentionLayer
import torch

if __name__ == "__main__":
    input = torch.randn(32, 128, 64, 64) # (b, c, h, w)
    fca_layer = MultiSpectralAttentionLayer(channel = 128, dct_h = 64, dct_w = 64, reduction = 16, freq_sel_method = 'top16')
    output = fca_layer(input)
    print(output.shape)
    
```
***

### 39. Attention Augmented Convolutional Networks Usage

#### 39.1. Paper

[Attention Augmented Convolutional Networks (ICCV 2019)](https://arxiv.org/abs/1904.09925)

#### 39.2. Overview

![](.//AttentionMechanism/model/img/AAAttention.png)

#### 39.3. Usage Code

```python
from AttentionMechanism.model.attention.AAAttention import AugmentedConv
import torch

if __name__ == "__main__":
    input = torch.randn((16, 3, 32, 32))
    augmented_conv = AugmentedConv(in_channels=3, out_channels=64, kernel_size=3, dk=40, dv=4, Nh=4, relative=True, stride=2, shape=16)
    output = augmented_conv(input)
    print(output.shape)
    
```
***

### 40. Global Context Attention Usage

#### 40.1. Paper

[GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond (ICCVW 2019 Best Paper)](https://arxiv.org/abs/1904.11492)

[Global Context Networks (TPAMI 2020)](https://arxiv.org/abs/2012.13375)

#### 40.2. Overview

![](.//AttentionMechanism/model/img/GCNet.png)

#### 40.3. Usage Code

```python
from AttentionMechanism.model.attention.GCAttention import GCModule
import torch

if __name__ == "__main__":
    input = torch.randn(16, 64, 32, 32)
    gc_layer = GCModule(64)
    output = gc_layer(input)
    print(output.shape)
    
```
***

### 41. Linear Context Transform Attention Usage

#### 41.1. Paper

[Linear Context Transform Block (AAAI 2020)](https://arxiv.org/pdf/1909.03834v2)

#### 41.2. Overview

![](.//AttentionMechanism/model/img/LCTAttention.png)

#### 41.3. Usage Code

```python
from AttentionMechanism.model.attention.LCTAttention import LCT
import torch

if __name__ == "__main__":
    x = torch.randn(16, 64, 32, 32)
    attn = LCT(64, 8)
    y = attn(x)
    print(y.shape)
    
```
***

### 42. Gated Channel Transformation Usage

#### 42.1. Paper

[Gated Channel Transformation for Visual Recognition (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Gated_Channel_Transformation_for_Visual_Recognition_CVPR_2020_paper.pdf)

#### 42.2. Overview

![](.//AttentionMechanism/model/img/GCT.png)

#### 42.3. Usage Code

```python
from AttentionMechanism.model.attention.GCTAttention import GCT
import torch

if __name__ == "__main__":
    input = torch.randn(16, 64, 32, 32)
    gct_layer = GCT(64)
    output = gct_layer(input)
    print(output.shape)
    
```
***

### 43. Gaussian Context Attention Usage

#### 43.1. Paper

[Gaussian Context Transformer (CVPR 2021)](https://openaccess.thecvf.com//content/CVPR2021/papers/Ruan_Gaussian_Context_Transformer_CVPR_2021_paper.pdf)

#### 43.2. Overview

![](.//AttentionMechanism/model/img/GaussianCA.png)

#### 43.3. Usage Code

```python
from AttentionMechanism.model.attention.GaussianAttention import GCA
import torch

if __name__ == "__main__":
    input = torch.randn(16, 64, 32, 32)
    gca_layer = GCA(64)
    output = gca_layer(input)
    print(output.shape)
    
```

***



# MLP Series

- Pytorch implementation of ["RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition---arXiv 2021.05.05"](https://arxiv.org/pdf/2105.01883v1.pdf)

- Pytorch implementation of ["MLP-Mixer: An all-MLP Architecture for Vision---arXiv 2021.05.17"](https://arxiv.org/pdf/2105.01601.pdf)

- Pytorch implementation of ["ResMLP: Feedforward networks for image classification with data-efficient training---arXiv 2021.05.07"](https://arxiv.org/pdf/2105.03404.pdf)

- Pytorch implementation of ["Pay Attention to MLPs---arXiv 2021.05.17"](https://arxiv.org/abs/2105.08050)


- Pytorch implementation of ["Sparse MLP for Image Recognition: Is Self-Attention Really Necessary?---arXiv 2021.09.12"](https://arxiv.org/abs/2109.05422)

### 1. RepMLP Usage
#### 1.1. Paper
["RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition"](https://arxiv.org/pdf/2105.01883v1.pdf)

#### 1.2. Overview
![](./model/img/repmlp.png)

#### 1.3. Usage Code
```python
from fightingcv_attention.mlp.repmlp import RepMLP
import torch
from torch import nn

N=4 #batch size
C=512 #input dim
O=1024 #output dim
H=14 #image height
W=14 #image width
h=7 #patch height
w=7 #patch width
fc1_fc2_reduction=1 #reduction ratio
fc3_groups=8 # groups
repconv_kernels=[1,3,5,7] #kernel list
repmlp=RepMLP(C,O,H,W,h,w,fc1_fc2_reduction,fc3_groups,repconv_kernels=repconv_kernels)
x=torch.randn(N,C,H,W)
repmlp.eval()
for module in repmlp.modules():
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        nn.init.uniform_(module.running_mean, 0, 0.1)
        nn.init.uniform_(module.running_var, 0, 0.1)
        nn.init.uniform_(module.weight, 0, 0.1)
        nn.init.uniform_(module.bias, 0, 0.1)

#training result
out=repmlp(x)
#inference result
repmlp.switch_to_deploy()
deployout = repmlp(x)

print(((deployout-out)**2).sum())
```

### 2. MLP-Mixer Usage
#### 2.1. Paper
["MLP-Mixer: An all-MLP Architecture for Vision"](https://arxiv.org/pdf/2105.01601.pdf)

#### 2.2. Overview
![](./model/img/mlpmixer.png)

#### 2.3. Usage Code
```python
from fightingcv_attention.mlp.mlp_mixer import MlpMixer
import torch
mlp_mixer=MlpMixer(num_classes=1000,num_blocks=10,patch_size=10,tokens_hidden_dim=32,channels_hidden_dim=1024,tokens_mlp_dim=16,channels_mlp_dim=1024)
input=torch.randn(50,3,40,40)
output=mlp_mixer(input)
print(output.shape)
```

***

### 3. ResMLP Usage
#### 3.1. Paper
["ResMLP: Feedforward networks for image classification with data-efficient training"](https://arxiv.org/pdf/2105.03404.pdf)

#### 3.2. Overview
![](./model/img/resmlp.png)

#### 3.3. Usage Code
```python
from fightingcv_attention.mlp.resmlp import ResMLP
import torch

input=torch.randn(50,3,14,14)
resmlp=ResMLP(dim=128,image_size=14,patch_size=7,class_num=1000)
out=resmlp(input)
print(out.shape) #the last dimention is class_num
```

***

### 4. gMLP Usage
#### 4.1. Paper
["Pay Attention to MLPs"](https://arxiv.org/abs/2105.08050)

#### 4.2. Overview
![](./model/img/gMLP.jpg)

#### 4.3. Usage Code
```python
from fightingcv_attention.mlp.g_mlp import gMLP
import torch

num_tokens=10000
bs=50
len_sen=49
num_layers=6
input=torch.randint(num_tokens,(bs,len_sen)) #bs,len_sen
gmlp = gMLP(num_tokens=num_tokens,len_sen=len_sen,dim=512,d_ff=1024)
output=gmlp(input)
print(output.shape)
```

***

### 5. sMLP Usage
#### 5.1. Paper
["Sparse MLP for Image Recognition: Is Self-Attention Really Necessary?"](https://arxiv.org/abs/2109.05422)

#### 5.2. Overview
![](./model/img/sMLP.jpg)

#### 5.3. Usage Code
```python
from fightingcv_attention.mlp.sMLP_block import sMLPBlock
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,3,224,224)
    smlp=sMLPBlock(h=224,w=224)
    out=smlp(input)
    print(out.shape)
```

### 6. vip-mlp Usage
#### 6.1. Paper
["Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition"](https://arxiv.org/abs/2106.12368)

#### 6.2. Usage Code
```python
from fightingcv_attention.mlp.vip-mlp import VisionPermutator
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = VisionPermutator(
        layers=[4, 3, 8, 3], 
        embed_dims=[384, 384, 384, 384], 
        patch_size=14, 
        transitions=[False, False, False, False],
        segment_dim=[16, 16, 16, 16], 
        mlp_ratios=[3, 3, 3, 3], 
        mlp_fn=WeightedPermuteMLP
    )
    output=model(input)
    print(output.shape)
```




# Acknowledgements
During the development of this project, the following open-source projects provided significant help and support. We hereby express our sincere gratitude:

- [**https://github.com/xmu-xiaoma666/External-Attention-pytorch**](https://github.com/xmu-xiaoma666/External-Attention-pytorch)

- [**https://github.com/cmhungsteve/Awesome-Transformer-Attention**](https://github.com/cmhungsteve/Awesome-Transformer-Attention)


