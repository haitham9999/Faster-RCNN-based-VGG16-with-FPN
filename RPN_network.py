import torch.nn as nn
import torch
import torch.nn.functional as F
import time
import torchvision
import anchors_assigning
from encode_decode import BoxCoder
import numpy as np
from sklearn.utils import shuffle
class RPN(nn.Module):
    
    def __init__(self, Anchors_generation, box_coder, RPN_HEAD):
        super(RPN, self).__init__()   

        self.Anchors_generation = Anchors_generation
        self.rpn_head = RPN_HEAD
        self.box_coder = box_coder


    def forward(self, feature_map,  targets=None):
        

        features = [ feats[0] for feats in feature_map ]
       
        anchors = self.Anchors_generation(  [features[0]]  )

        rpn_output = self.rpn_head(feature_map )

        choosing_anchors = torch.where(  (anchors[:,0] >= 0) & (anchors[:,1] >= 0)
                                            & (anchors[:,2]<=1024) & ( anchors[:,3]<=800) )[0]
        anchors = anchors[choosing_anchors]

   
        
        if self.training:
            
            
            ''' Bounding box labels assigning '''
            encoded_assigned_bbox_list, labels_list = [], []
            for box in targets:
                assigned_bbox = anchors_assigning.assign_anchors_to_BBoxes( anchors.unsqueeze(0), 
                                                            box['boxes'].unsqueeze(0) )
                labels_list.append(assigned_bbox['labels'])
                encoded_boxes = self.box_coder.encode( [assigned_bbox['bboxes'] ], [anchors] )
                encoded_assigned_bbox_list.append(encoded_boxes[0])
            '''End '''
            ''' picking up loss-contributed positives and negatives instances '''
            positive_indices, negative_indices = [], []
           
            for labels_info in labels_list:
                positive_indices.append( torch.where(labels_info > 0) [0] )
                negative_indices.append( torch.where(labels_info == 0)[0] )
            Final_contributed_instances , Final_contributed_positive_instances = [], []

            for indix, (positives, negatives) in enumerate(zip(positive_indices, negative_indices)):
                if not indix:
                    if len(positives) > 16:
                      positives = shuffle(positives, replace=False)
                      choosing_positives = positives[:16]
                      Final_contributed_positive_instances.append(choosing_positives)
                      
                      negatives = shuffle(negatives, replace=False)
                      choosing_negatives = negatives[:256]
                      Final_contributed_instances.append( torch.cat( (choosing_positives, choosing_negatives) ))
                    
                    else:
                      choosing_positives = positives
                      Final_contributed_positive_instances.append(choosing_positives)

                      negatives = shuffle(negatives, replace=False)
                      choosing_negatives = negatives[ : ( 256 + ( 16-positives.numel()) ) ]
                
                      Final_contributed_instances.append( torch.cat( (choosing_positives, choosing_negatives) ))
                else:
                      if len(positives) > 16:
                        choosing_positives = positives[:16] + len(labels_list[indix])
                        Final_contributed_positive_instances.append(choosing_positives)

                        choosing_negatives = negatives[:256] + len(labels_list[indix])
                        Final_contributed_instances.append( torch.cat( (choosing_positives, choosing_negatives) ))
                      
                      else:
                        choosing_positives = positives + len(labels_list[indix])
                        Final_contributed_positive_instances.append(choosing_positives)
                        choosing_negatives = negatives[ : ( 256 + ( 16-positives.numel()) ) ] + len(labels_list[indix])
                  
                        Final_contributed_instances.append( torch.cat( (choosing_positives, choosing_negatives) ))
            ''' End '''
            ''' predicted data '''
            
            _pred_anchor_locs = rpn_output['pred_anchor_locs_all'][:,choosing_anchors].view(-1,4)[torch.cat(Final_contributed_positive_instances)]
            _pred_cls_scores = rpn_output['pred_cls_scores_all'][:,choosing_anchors].view(-1)[torch.cat(Final_contributed_instances)]
           
            ''' End ''' 
            ''' Targeted data '''
           
            _targeted_anchor_locs = torch.cat( encoded_assigned_bbox_list ) [torch.cat(Final_contributed_positive_instances)]
            _targeted_cls_scores = torch.cat( labels_list )[ torch.cat(Final_contributed_instances) ]
           
            ''' Loss Calculation '''
           
            rpn_loss_all, loc_loss, rpn_cls_loss_all = self.Loss(_targeted_anchor_locs ,_pred_anchor_locs, _targeted_cls_scores, _pred_cls_scores  )
            ''' End ''' 
            
          
            ''' proposals candidate '''
          
            
            proposals_info = self.choosing_proposals( rpn_output['pred_anchor_locs_all'][:,choosing_anchors].detach(), anchors , targets, rpn_output['pred_cls_scores_all'][:,choosing_anchors].detach())
         
            ''' End '''
            return {'rpn_loss_all':rpn_loss_all, 'loc_loss':loc_loss, 'rpn_cls_loss_all':rpn_cls_loss_all, 'proposals_reference':proposals_info['proposals_reference'],
                  
                       'proposals_bbox':proposals_info['proposals_bbox'], 'proposals_labels':proposals_info['proposals_labels'], 
                      '_targeted_cls_scores':labels_list,'anchors':anchors, 'Final_contributed_positive_instances':Final_contributed_positive_instances,
                       'Final_contributed_instances':Final_contributed_instances,
                       'false_true_list':proposals_info['false_true_list'],'false_true_list_bbox':proposals_info['false_true_list_bbox']}
            
        else:
            
            return self.choosing_proposals_for_testing(rpn_output['pred_anchor_locs_all'][:,choosing_anchors], anchors , rpn_output['pred_cls_scores_all'][:,choosing_anchors])



    
    def choosing_proposals(self, pred_anchor_locs, anchors,  targets, pred_cls_scores_all):

            ''' choosing proposals boxes '''
            calc_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) ,
                                (boxes[:, 3] - boxes[:, 1]))
        
            
            proposals_bbox, proposals_labels, proposals_reference, false_true_list, false_true_list_bbox = [], [], [], [], []


            proposals = self.box_coder.decode( pred_anchor_locs.view(-1,4), 
                                                     [ torch.cat((anchors, anchors)) ] )

            proposals = proposals.view(pred_anchor_locs.shape[0],-1,4)

            assigned_bboxes = [ anchors_assigning.assign_anchors_to_BBoxes( proposal.unsqueeze(0), 
                                                box['boxes'].unsqueeze(0) ) for (proposal,  box) in zip(proposals, targets)]

            proposals_offsets = self.box_coder.encode( [ torch.cat((assigned_bboxes[0]['bboxes'],assigned_bboxes[1]['bboxes'])) ], 
                                                [proposals.view(-1,4) ] )[0].view(pred_anchor_locs.shape[0],-1,4)




            for indix, (assigned_bbox, proposals_offset) in enumerate(zip(assigned_bboxes, proposals_offsets)):
                
                positive_indices = torch.where( assigned_bbox['labels'] > 0 )[0]

                negative_indices = torch.where(assigned_bbox['labels'] == 0)[0]
                positive_indices = shuffle(positive_indices, replace=False)
                negative_indices = shuffle(negative_indices, replace=False)
                
            

                false_true = torch.sort( pred_cls_scores_all[indix][negative_indices].view(-1) , descending=True )
                false_true_list.append(false_true)
                false_true_list_bbox.append( proposals[indix][false_true[1][:100]])
                areas = calc_area( anchors[negative_indices] )
                negative_proposals_threshold_by_scores = torch.where(  
                  ( proposals[indix][negative_indices][:,0] >= 0 ) & (proposals[indix][negative_indices][:,1] >= 0)
                                          & (proposals[indix][negative_indices][:,2]<=1024) & ( proposals[indix][negative_indices][:,3]<=800)&
                                          (areas[0] >= 20) & (areas[1] >= 20) )[0]
                
                negative_proposals_threshold_by_scores = negative_indices[negative_proposals_threshold_by_scores]
                negative_proposals_threshold_by_scores = shuffle(negative_proposals_threshold_by_scores, replace=False)
            

                if positive_indices.numel()>16:
                    final_positive_indices = positive_indices[:16]
                    positive_proposals = proposals[indix][final_positive_indices]
                    positive_proposal_reference = proposals_offset[ final_positive_indices ]
                    positive_labels = assigned_bbox['labels'][final_positive_indices]
                
                    negative_proposals = proposals[indix][negative_proposals_threshold_by_scores[:128]]
                    negative_proposals_reference = proposals_offset[negative_proposals_threshold_by_scores[:128]]
                    negative_labels = assigned_bbox['labels'][ negative_proposals_threshold_by_scores[:128] ]
                   
                else:
                    final_positive_indices = positive_indices
                    positive_proposals = proposals[indix][final_positive_indices]
                    positive_proposal_reference = proposals_offset[ final_positive_indices ]
                    positive_labels = assigned_bbox['labels'][final_positive_indices]
                  

                    gap = 16 - final_positive_indices.numel()
                    negative_proposals = proposals[indix][negative_proposals_threshold_by_scores[:128 + gap]]
                    negative_proposals_reference = proposals_offset[negative_proposals_threshold_by_scores[:128 + gap]]
                    negative_labels = assigned_bbox['labels'][negative_proposals_threshold_by_scores[:128 + gap]]
               
                  
                                         
                                                            
        
                proposals_labels.append(torch.cat((positive_labels, negative_labels)))
                proposals_bbox.append(torch.cat((positive_proposals, negative_proposals)))
                
                proposals_reference.append(torch.cat((positive_proposal_reference, negative_proposals_reference)))

            return { 'proposals_bbox':proposals_bbox, 'proposals_labels':proposals_labels, 
            'proposals_reference':proposals_reference, 'false_true_list':false_true_list,'false_true_list_bbox':false_true_list_bbox  }


    def choosing_proposals_for_testing(self, pred_anchor_locs, anchors, pred_cls_scores):
        ''' choosing proposals boxes '''
        proposals_bbox = []
        all_scores = []


        proposals = self.box_coder.decode( pred_anchor_locs.view(-1,4), 
                                                           [ torch.cat(  ((anchors),)*len(pred_anchor_locs)    ) ] )

        proposals = proposals.view(pred_anchor_locs.shape[0],-1,4)
        print(pred_anchor_locs.shape[0])
        print(proposals.shape)


        for indix, pred_scores in enumerate(pred_cls_scores):
                           
        
            
            #scores_indices = torch.sort( pred_scores.view(-1),  descending=True)[1]


            greater_than_zero = torch.where(pred_scores.view(-1) > 0)[0]

            print('greater_than_zero', len(greater_than_zero))

            sigmoid = nn.Sigmoid()

            pred_scores = sigmoid( pred_scores.view(-1)[greater_than_zero] )
            picked_proposals = proposals[indix][greater_than_zero]

            indices = torchvision.ops.nms( picked_proposals, pred_scores , iou_threshold=0.5  )


   
            all_scores.append( pred_scores[indices[:256]] )
            

            proposals_bbox.append( picked_proposals[indices[:256]] )

    
        return  { 'proposals_bbox':proposals_bbox, 'scores':all_scores  }



    def Loss(self, gt_bbox, pred_bbox , gt_label,pred_label ):
      
        classifer = F.binary_cross_entropy_with_logits( pred_label , gt_label.type(torch.float) )


        bbox_regression = F.smooth_l1_loss( pred_bbox ,gt_bbox , beta=1 / 9,
            reduction='sum') / (len(gt_label) )

   
        loss = bbox_regression * 10  + classifer

        return loss, bbox_regression, classifer


    
              