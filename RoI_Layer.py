import torch.nn as nn
import torch

class ROIPoolling(nn.Module):
    def __init__(self, levels=[1]):
        super().__init__()
        
        self.levels = levels
        
    
    def generation_levels(self, rios):
        levels_assignment = [ [ 2100 ], [ 2100 , 10575], [ 10575  ] ]
        calc_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * 
                                  (boxes[:, 3] - boxes[:, 1]))
        areas = calc_area(rios)
        levels = []
        levels.append( torch.where( areas <= 2100)[0] )

        levels.append( torch.where(  (areas < 10575) & (areas > 2100)  )[0] )

        levels.append( torch.where( areas >= 10575)[0] )
        

        return levels

    def forward(self, features, roiss):
        final_output = []
        images_nums = len(features)
        res = []
        image_extraction = [] 
        len_rois = roiss[0].shape[0]
        rois_leveled_indices = []
        for position, rois in enumerate(roiss):
         
        
          rois_levels = self.generation_levels(rois)
          rois_leveled_indices.append( torch.cat(  rois_levels   ) + (  position *  (len(rois) - 1) ))
          
          level_res = []
          for index, feat_size_level in enumerate(rois_levels):

        

            if not len(feat_size_level) == 0:

                feat = features[index][position]
                h, w = feat.shape[1:]

                picked_rois = rois[feat_size_level] / torch.tensor((1024, 800, 1024, 800), device = rois.device)
                
                n = len(picked_rois)
                x1 = picked_rois[:,0]
                y1 = picked_rois[:,1]
                x2 = picked_rois[:,2]
                y2 = picked_rois[:,3]

                x1 = torch.clamp( torch.floor(x1 * w).type(torch.int), 0)
                x2 = torch.clamp( torch.ceil(x2 * w).type(torch.int), 0)
                
                y1 = torch.clamp( torch.floor(y1 * h).type(torch.int), 0)
                y2 = torch.clamp( torch.ceil(y2 * h).type(torch.int), 0)
                
                
                for output_shape in self.levels:
                  output_shape = (output_shape, output_shape)
                  maxpool = nn.AdaptiveMaxPool2d(output_shape)
                  
                  for i in range(n):
            
                      

                      if x1[i] == 0 and x2[i] == 0:
                        x2[i] = x2[i]+1
                      if y1[i] == 0 and y2[i] == 0:
                        y2[i] = y2[i]+1

                      if x1[i] >= w:
                        x1[i] = torch.tensor((w-1)).type(torch.int)
                      if y1[i] >= h:
                        y1[i] = torch.tensor((h-1)).type(torch.int)

                      
                      img = maxpool(feat[ :, y1[i]:y2[i], x1[i]:x2[i]])
                      level_res.append(img)
          
          image_extraction.append(torch.stack(level_res,dim=0).view( len(rois) , -1))
          

        
         
     
        return image_extraction, torch.cat(rois_leveled_indices)