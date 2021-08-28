import torch
def assign_anchors_to_BBoxes(anchors, gt_bboxes):
    all_labels, all_gt_bboxes = [], []
    for anchor_batch, gt_bbox_batch in zip(anchors, gt_bboxes):

        anchors_over_bboxes_IoU = calc_IoU(anchor_batch, gt_bbox_batch[:,1:])

        
        labels = torch.empty( len(anchor_batch), dtype=torch.long, device=anchor_batch.device).fill_(-1)

        assigned_boxes = torch.zeros_like( anchor_batch , dtype=anchor_batch.dtype,
                                                            device=anchor_batch.device )

        max_ious, indices = torch.max(anchors_over_bboxes_IoU, dim=1)
        arrow_indices = torch.arange(len(indices), device = anchors.device)
        
        indices_iou_threshold_positive = torch.where(max_ious >= 0.7)[0]
        bounding_boxes_satisifed_the_threshold_condition = torch.unique(indices[indices_iou_threshold_positive])

        
        corresponding_positive_bbox = arrow_indices[indices_iou_threshold_positive]

        
        assigned_boxes[corresponding_positive_bbox] = gt_bbox_batch[:,1:][indices[indices_iou_threshold_positive]]

        labels[corresponding_positive_bbox] = gt_bbox_batch[:,0][indices[indices_iou_threshold_positive]].type(torch.long) + 1
        
    
        indices_iou_threshold_negative = torch.where(max_ious <= 0.3)[0]
        

        corresponding_negative_bbox = arrow_indices[indices_iou_threshold_negative]

        labels[corresponding_negative_bbox] = 0

        all_bounding_boxes_indices = torch.arange(len(gt_bbox_batch), device=bounding_boxes_satisifed_the_threshold_condition.device)
      
        missing_bboxes = [ not i in bounding_boxes_satisifed_the_threshold_condition  for  i in all_bounding_boxes_indices]

        indices_of_missing_bboxes = all_bounding_boxes_indices[missing_bboxes]

        if  indices_of_missing_bboxes.numel():
            for missed_bbx in indices_of_missing_bboxes:

                values, _indices = torch.max(anchors_over_bboxes_IoU[:,missed_bbx] , dim=0)
                if values >= 0.5:
                        assigned_boxes[ _indices ] = gt_bbox_batch[missed_bbx][1:]
                        labels[_indices] = gt_bbox_batch[:,0][missed_bbx].type(torch.int64) + 1



        

        all_labels.append(labels)
        all_gt_bboxes.append(assigned_boxes)
        
  
    return { 'labels':torch.cat(all_labels), 'bboxes':torch.cat(all_gt_bboxes) }
		
		
def calc_IoU(anchor_batch, gt_bbox_batch):

    calc_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * 
                                (boxes[:, 3] - boxes[:, 1]))
    achors_area = calc_area(anchor_batch)
    gt_area = calc_area(gt_bbox_batch)

    upper_left_intersection = torch.max(  anchor_batch[:, None, :2] , gt_bbox_batch[:,:2]   )
    lower_right_intersection = torch.min(  anchor_batch[:, None, 2:] , gt_bbox_batch[:,2:]   )

    intersection_dimensions = (lower_right_intersection - upper_left_intersection).clamp(min=0)
    
    intersection_area = intersection_dimensions[:, :, 0] * intersection_dimensions[:, :, 1]

    union_areas = achors_area[:, None] + gt_area - intersection_area 
    
    return intersection_area/union_areas 