import torch.nn as nn
import torch
import time
class Faster_RCNN(nn.Module):
	def __init__(self, backbone, rpn, fast_rcnn ):
		super(Faster_RCNN, self).__init__()
		self.backbone = backbone
		self.rpn = rpn
		self.fast_rcnn = fast_rcnn
		
	def forward(self,images, targets = None ):
    
		if self.training:
				
				features = self.backbone(torch.stack(images))
				''' RPN output content {'rpn_loss_all':rpn_loss_all, 'loc_loss':loc_loss, 
                    'rpn_cls_loss_all':rpn_cls_loss_all, 'proposals_labels':proposals_info['proposals_labels'],
                      'proposals_bbox':proposals_info['proposals_bbox'], 'proposals_reference':proposals_info['proposals_reference']}
        '''
				image_size = images[0].shape[1:3]
				output = self.rpn( features, targets )
				
				
				fast_RCNN_output = self.fast_rcnn(features, output  )
	
				BOth_network_total_loss = output['rpn_loss_all'] + fast_RCNN_output['loss']
				RPN_total_loss          = output['rpn_loss_all']
				RPN_cls_loss            = output['rpn_cls_loss_all']
				RPN_locs_loss           = output['loc_loss']
        
				Fast_total_loss          = fast_RCNN_output['loss']
				Fast_cls_loss            = fast_RCNN_output['loss_sc']
				Fast_locs_loss           = fast_RCNN_output['bbox_regression_loss']


				return { 'BOth_network_total_loss':BOth_network_total_loss, 'RPN_total_loss':RPN_total_loss, 'RPN_cls_loss':RPN_cls_loss, 'RPN_locs_loss':RPN_locs_loss,
                    'proposals_bbox':output['proposals_bbox'], 'proposals_labels':output['proposals_labels'] ,
                    'anchors':output['anchors'], 'Final_contributed_positive_instances':output['Final_contributed_positive_instances'],
                       'Final_contributed_instances':output['Final_contributed_instances'],
                       'false_true_list':output['false_true_list'],'false_true_list_bbox':output['false_true_list_bbox'], 'Fast_total_loss':Fast_total_loss,
                          'Fast_cls_loss':Fast_cls_loss, 'Fast_locs_loss':Fast_locs_loss,
                        'bboxes_fast':fast_RCNN_output['bboxes'], 'fast_scores':fast_RCNN_output['scores']  }
        
		else:
			  features = self.backbone(torch.stack(images))
			  ''' RPN output content {'rpn_loss_all':rpn_loss_all, 'loc_loss':loc_loss, 
                    'rpn_cls_loss_all':rpn_cls_loss_all, 'proposals_labels':proposals_info['proposals_labels'],
                      'proposals_bbox':proposals_info['proposals_bbox'], 'proposals_reference':proposals_info['proposals_reference']}
			  '''
			  image_size = images[0].shape[1:3]
			
			  output = self.rpn( features )
			  ''' {'loss':loss, 'loss_sc':loss_sc, 'bbox_regression_loss':bbox_regression_loss, 'scores':roi_scores, 'bboxes': output_re_transformmed_bbox} '''
			  
			  fast_RCNN_output = self.fast_rcnn( features, output  )
                                              
        

        
			  return fast_RCNN_output













