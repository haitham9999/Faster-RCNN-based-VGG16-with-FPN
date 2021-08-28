import torch.nn.functional as F
from torch import nn 
import torchvision
import torch

class Main(nn.Module):
  def __init__(self, num = []):
    super(Main, self).__init__()
    self.num = num
    self.model = nn.Sequential(*list(torchvision.models.vgg16_bn(pretrained=True).features.children())[:-7])
    for i, param in enumerate(self.model.parameters()):
        param.requires_grad = False
    

  def forward(self, x):
    outputList = {}
    for i, layer in enumerate(self.model):
        x = layer(x)
        if i in self.num:
          outputList[i] = x
    return outputList
	
	
	

class FPN(nn.Module):

  def __init__(self, in_channels_list = [], out_channels = 0):
    super(FPN, self).__init__()
    self.inner_blocks = nn.ModuleList()
    self.layer_blocks = nn.ModuleList()
    for in_channels in [256, 512, 512]:
      if in_channels == 0:
          raise ValueError("in_channels=0 is currently not supported")
      inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
      layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
      self.inner_blocks.append(inner_block_module)
      self.layer_blocks.append(layer_block_module)

    for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

  def forward(self, x):
    
    names = list( x.keys() )
    x = list( x.values() )
    last_feature = x[-1]
    last_inner = self.get_output_from_inner_layer(last_feature, -1)
 
    results = []
    results.append(self.get_output_from_outer_layer(last_inner, -1))

    for num in range(len(x)-2, -1, -1):

      inner_lateral = self.get_output_from_inner_layer(x[num], num)
      feat_shape = inner_lateral.shape[-2:]
      inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
      last_feature = inner_lateral + inner_top_down
      
      results.insert(0, self.get_output_from_outer_layer(last_feature, num))
   
    
    return results

  def get_output_from_inner_layer(self, x, num):
    output = self.inner_blocks[num](x)
    return output

  def get_output_from_outer_layer(self,x, num):
    output = self.layer_blocks[num](x)
    return output



class FPN_initializer(nn.Module):
  def __init__(self, levels = [22, 32, 36]  ):
    super(FPN_initializer, self).__init__()
    levels = levels
    self.backbone = Main(levels)
    self.fbn = FPN(levels, 256)

  def forward(self, x):
      x  = self.backbone(x)

     
      x = self.fbn(x)
      return x

