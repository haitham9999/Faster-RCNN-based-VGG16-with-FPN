import torch
from torch import nn

class Generate_anchors(torch.nn.Module):
  def __init__(self, device = torch.device('cpu'), sizes = [[16, 55, 85],
                   [ 128, 170 ,200], [256, 370 ,512]]):
    super(Generate_anchors, self).__init__()
    self.device = device
    
    aspect_ratios = [[0.5, 1, 2],] * len(sizes)
    self.cell_anchors = [self._generate_anchors(size, aspect_ratio, device = self.device )
                          for size, aspect_ratio in zip(sizes, aspect_ratios)]


  def _generate_anchors(self, scales, aspect_ratios, dtype: torch.dtype = torch.float32,
                        device: torch.device = torch.device('cpu') ):

      scales = torch.as_tensor(scales, dtype=dtype, device=device)
      aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
      h_ratios = torch.sqrt(aspect_ratios)
      w_ratios = 1 / h_ratios

      ws = (w_ratios[:, None] * scales[None, :]).view(-1)
      hs = (h_ratios[:, None] * scales[None, :]).view(-1)

      base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
     
      return base_anchors.round()

  def grid_anchors(self, grid_sizes, strides) :
        anchors = []
        cell_anchors = self.cell_anchors
       

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors ):  
            
       
            grid_height, grid_width = size
            stride_height, stride_width = stride
            
            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = torch.arange( 0, grid_width, dtype=torch.float32, device=self.device ) * stride_width
                                       
           
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=self.device ) * stride_height
                
            

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
           
            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append( (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
                
            
  
              
        return anchors

  def forward(self, feature_maps ):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]

        image_size = [800, 1024]
     
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
    
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        
    
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)

        anchors_over_all_feature_maps = torch.cat(anchors_over_all_feature_maps)
        
        
        return anchors_over_all_feature_maps







