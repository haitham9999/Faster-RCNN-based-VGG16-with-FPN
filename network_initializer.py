from VGG_16_with_FPN import FPN_initializer
from RPN_network import RPN
import torch 
from fast_rcnn import Fast_RCNN
from faster_rcnn import Faster_RCNN
from generate_anchors import Generate_anchors  
from encode_decode import BoxCoder
from rpn_network_head import rpn_head

class Network_Initializer(Faster_RCNN):
	
	def __init__(self, in_channels = 256 , level = [7], 
					mid_channels=256, num_anchors=9, device =torch.device('cpu'),
          sizes = [[16, 55, 85], [ 128, 170 ,200], [256, 370 ,512]]):
		
	
  
		backbone = FPN_initializer()
		
		Anchors_generation = Generate_anchors( device = device , sizes = sizes)
  

		box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

		RPN_HEAD = rpn_head( in_channels=in_channels, mid_channels=mid_channels , num_anchors = num_anchors)

		rpn = RPN ( Anchors_generation, box_coder, RPN_HEAD )
		
		fast_rcnn = Fast_RCNN( in_channels = in_channels , level = level )
		
		super(Network_Initializer, self).__init__(backbone, rpn, fast_rcnn)