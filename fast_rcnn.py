import torch.nn as nn
import torch
from encode_decode import BoxCoder
from RoI_Layer import ROIPoolling
import torch.nn.functional as F

class Fast_RCNN(nn.Module):
    def __init__(self , in_channels = 256 , level = [7] ):
        super().__init__()
    
        self.sppnet = ROIPoolling(level)
    
        feature_dim = level[0] * level[0] * in_channels

        self.FC1 = nn.Linear(in_features=feature_dim, out_features=1024)
        self.FC2 = nn.Linear(in_features=1024, out_features=1024)
    
        self.cls_score = nn.Linear(1024, 2)
        self.bbox = nn.Linear(1024, 8)

        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def forward(self, feature_map, proposal_data):
        
     
        if self.training:
            proposals_bbox, proposals_labels = proposal_data['proposals_bbox'], proposal_data['proposals_labels']
            
            spp_output, levels = self.sppnet(feature_map, proposals_bbox)

            
                
            feat = F.relu( self.FC1( torch.cat(spp_output) ) )
            feat = F.relu( self.FC2 (feat) )
            roi_scores = self.cls_score(feat)
            roi_predicted_bbox = self.bbox(feat).view(-1, 2 ,4)
            
            ''' calc loss '''
            loss, loss_sc, bbox_regression_loss = self.calc_loss(roi_scores, roi_predicted_bbox,
                                                      proposals_labels, proposal_data['proposals_reference'], levels)
            
        
            
            output_re_transformmed_bbox = []

            boxes = self.box_coder.decode( roi_predicted_bbox.detach()[:,1]   , [ torch.cat(proposals_bbox)[levels]] ).view( len(proposals_bbox), -1, 4)
                                                              

         
            
            #return {'loss':loss, 'loss_sc':loss_sc, 'bbox_regression_loss':bbox_regression_loss, 'scores':roi_scores, 'bboxes': output_re_transformmed_bbox}
            
            return {'loss':loss, 'loss_sc':loss_sc, 'bbox_regression_loss':bbox_regression_loss, 'scores':roi_scores.view( len(proposals_bbox),-1, 2), 'bboxes': boxes}
        
        else:
            
            roi_scores, roi_predicted_bbox = [] , []

          
            proposals_bbox     = proposal_data['proposals_bbox']
            

            spp_output, levels = self.sppnet(feature_map, proposals_bbox)

            
                
            feat               = F.relu( self.FC1( torch.cat(spp_output) ) )
            feat               = F.relu( self.FC2 (feat) )
            roi_scores         = self.cls_score(feat)
            roi_predicted_bbox = self.bbox(feat).view(-1, 2 ,4)

            boxes = self.box_coder.decode( roi_predicted_bbox.detach()[:,1]   , [ torch.cat(proposals_bbox)[levels]] ).view( len(proposals_bbox), -1, 4)
            
            return {'bboxes': boxes, 'roi_scores':roi_scores.view( len(proposals_bbox),-1, 2)}


    def calc_loss(self, roi_scores, roi_predicted_bbox, proposals_labels, proposals_reference, levels):
                                                  
        loss_list, loss_sc_list, bbox_regression_list = [], [], []
      

        stacked_labels = torch.cat(proposals_labels)[levels]

     

        positive_indices = torch.where( stacked_labels )[0]
        loss_sc =   F.cross_entropy(roi_scores, stacked_labels)

        loss_pred_box_contri = roi_predicted_bbox[ positive_indices , stacked_labels[positive_indices] ]

        loss_targeted_box_contri = torch.cat(proposals_reference)[levels]
        loss_targeted_box_contri = loss_targeted_box_contri[positive_indices]

       

        bbox_regression = F.smooth_l1_loss(loss_pred_box_contri ,loss_targeted_box_contri , beta=1 / 9,
                      reduction='sum') /stacked_labels.numel()
        
      
   
          
        loss = loss_sc + bbox_regression 


        return loss, loss_sc, bbox_regression   

    




