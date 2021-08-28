from tqdm import tqdm
from functools import partial
import torch
import numpy as np
tqdm = partial(tqdm, position=0, leave=True)
import time 
from PIL import Image

import matplotlib.pyplot as plt
import cv2

def train_epocs(model, optimizer, data_loader , device, epochs=10, training_state=True ):
  for epoch in range(epochs):
        
        if epoch==6:
          parameters = model.parameters()
          optimizer = torch.optim.SGD(parameters, lr=0.000002
                                , momentum=0.99)
        total = 0
        sum_loss = 0
        sum_loss_cls = 0
        sum_loss_loc = 0
        sum_loss_fast = 0
        sum_loss_score_fast = 0
        sum_loss_loc_fast = 0
        idx = 0
        
        number = 0
        j=0

        for data in tqdm(data_loader):

          optimizer.zero_grad()
          images = [ image.to(device) for image in data[0] ]
          targets = [  { k: v.to(device) for k,v in da.items()  }  for da in data[1]]
          num_batch =  len(images)
          ''' gradient tracking setting '''
          if not training_state:
            with torch.no_grad():
              model.eval()
              output = model(images, targets )
              
          else:
              #start = time.time()
              model.train()
              output = model(images, targets)
              #torch.cuda.current_stream().synchronize()

              #print('forward time', time.time()-start)

          ##### test 
          #return {'rpn_loss_all':rpn_loss_all, 'loc_loss':loc_loss, 'rpn_cls_loss_all':rpn_cls_loss_all, 'proposals_labels':proposals_info['proposals_labels'],
          #            'proposals_bbox':proposals_info['proposals_bbox'], 'proposals_reference':proposals_info['proposals_reference'] }
          '''
          positives = torch.where(output['proposals_labels'][0] > 0.5)[0]
         

          print('output[''_targeted_cls_scores'']',output['_targeted_cls_scores'])
          anchors = output['kuku']
          positives_indices = torch.where(output['_targeted_cls_scores'][0] > 0) [0]
          print('positives_indices', positives_indices)

          true_boxes = anchors[positives_indices]

          print('true_boxes', true_boxes)

          print('targets', targets[0]['boxes'])

          path = data[2][0]
          img = Image.open(path)
          img = np.asarray(img)
          img = cv2.resize(img, (1024, 800), interpolation = cv2.INTER_AREA)
          img2 = img.copy()
          for box in true_boxes:
              
              a = int(box[0])
              b = int(box[1])
              c = int(box[2])
              d = int(box[3])
              if a>c | b>d:
                continue
              cv2.rectangle(img2 , (a,b), (c,d), (255, 0, 0), 2)
          plt.imshow(img2)    
          break
          '''






          ''' parameters updating '''
          Final_loss = output['BOth_network_total_loss']
         
          count = 0
          before = []
          after = []
          
          if training_state == True: 
            #start = time.time()
           

            Final_loss.backward()
            optimizer.step()
            
            #torch.cuda.current_stream().synchronize()

            #print('backward time', time.time()-start)

          j += 1
          total += num_batch
          sum_loss += output['RPN_total_loss'].item()
          sum_loss_cls += output['RPN_cls_loss'].item()
          sum_loss_loc += output['RPN_locs_loss'].item()
          sum_loss_fast += output['Fast_total_loss'].item()
          sum_loss_score_fast += output['Fast_cls_loss'].item()
          sum_loss_loc_fast +=  output['Fast_locs_loss'].item()
          if j  % 100 ==0:
                train_loss = np.float32(sum_loss/np.float(total))
                train_loss_cls = sum_loss_cls/total
                train_loss_loc = sum_loss_loc/total
                train_fast_all = sum_loss_fast/total
                train_fast_cls = sum_loss_score_fast/total
                train_fast_loc = sum_loss_loc_fast/total
                #if (epoch+1) % 5 == 0:
                #    torch.save(model.state_dict(), './rpn_%s.pth'%epoch)
                print("train_loss %.4f loc_loss %.4f   cls_loss %.4f " % (train_loss, train_loss_loc , train_loss_cls))
                print("##############train_loss_Fast %.4f loc_loss_fast %.4f   cls_loss_fast %.4f " % (train_fast_all, train_fast_loc , train_fast_cls))

                total = 0.0
                sum_loss =0.0
                sum_loss_cls = 0.0
                sum_loss_loc = 0.0
                sum_loss_fast = 0.0
                sum_loss_score_fast = 0.0
                sum_loss_loc_fast = 0.0
         
        train_loss = sum_loss/total
        train_loss_cls = sum_loss_cls/total
        train_loss_loc = sum_loss_loc/total
        train_fast_all = sum_loss_fast/total
        train_fast_cls = sum_loss_score_fast/total
        train_fast_loc = sum_loss_loc_fast/total
        #if (epoch+1) % 5 == 0:
        #    torch.save(model.state_dict(), './rpn_%s.pth'%epoch)
        print("######  train_loss %.4f loc_loss %.4f  cls_loss %.4f " % (train_loss, train_loss_loc, train_loss_cls))
        print("####### train_loss_Fast %.4f loc_loss_fast %.4f   cls_loss_fast %.4f " % (train_fast_all, train_fast_loc , train_fast_cls))
        print(number)
        torch.save(model.state_dict(), '/content/gdrive/MyDrive/fast-rcnn-model-save/Faster_trial_version4.mdl')



  return model