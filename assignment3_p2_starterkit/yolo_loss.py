import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class YoloLoss(nn.Module):
    def __init__(self,S,B,l_coord,l_noobj):
        super(YoloLoss,self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.eps = 1e-20

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
    
    def get_class_prediction_loss(self, classes_pred, classes_target):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        Returns:
        class_loss : scalar
        """
        
        class_loss = sum(sum((classes_pred - classes_target)**2))
        
        return class_loss
    
    
    def get_regression_loss(self, box_pred_response, box_target_response):   
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 5)
        box_target_response : (tensor) size (-1, 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified 
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as
        Returns:
        reg_loss : scalar
        
        """
        reg_loss = 0.0
        x, y, w, h = box_pred_response[:, 0], box_pred_response[:, 1], box_pred_response[:, 2], box_pred_response[:, 3]
        xhat, yhat, what, hhat = box_target_response[:, 0], box_target_response[:, 1], box_target_response[:, 2], box_target_response[:, 3]
        reg_loss += sum((x - xhat)**2) + sum((y - yhat)**2)
        reg_loss += sum((torch.sqrt(w+self.eps) - torch.sqrt(what+self.eps))**2) + sum((torch.sqrt(h+self.eps) - torch.sqrt(hhat+self.eps))**2)
        return reg_loss
    
     
    def get_contain_conf_loss(self, box_pred_response, box_target_response_iou):
        """
        Parameters:
        box_pred_response : (tensor) size ( -1 , 5)
        box_target_response_iou : (tensor) size ( -1 , 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified 
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as
        Returns:
        contain_loss : scalar
        
        """
        target_iou = box_target_response_iou.detach()
        contain_loss = 0.0
        c = box_pred_response[:, 4]
        chat = target_iou[:, 4]
        contain_loss = sum((c - chat)**2)
        return contain_loss
    
    def get_no_object_loss(self, target_tensor, pred_tensor, no_object_mask):
        """
        Parameters:
        target_tensor : (tensor) size (batch_size, S , S, 30)
        pred_tensor : (tensor) size (batch_size, S , S, 30)
        no_object_mask : (tensor) size (batch_size, S , S, 30)

        Returns:
        no_object_loss : scalar

        Hints:
        1) Create a 2 tensors no_object_prediction and no_object_target which only have the 
        values which have no object. 
        2) Have another tensor no_object_prediction_mask of the same size such that 
        mask with respect to both confidences of bounding boxes set to 1. 
        3) Create 2 tensors which are extracted from no_object_prediction and no_object_target using
        the mask created above to find the loss. 
        """
        #masking out values not containing obj
        no_object_prediction = pred_tensor[no_object_mask]
        no_object_target = target_tensor[no_object_mask]
        no_object_prediction_mask = torch.zeros(no_object_prediction.shape, dtype=torch.bool)
        no_object_prediction_mask[4::pred_tensor.shape[-1]] = True
        no_object_prediction_mask[9::pred_tensor.shape[-1]] = True
        no_obj_class_prediction = no_object_prediction[no_object_prediction_mask]
        no_obj_class_target = no_object_target[no_object_prediction_mask]
        no_object_loss = sum((no_obj_class_prediction - no_obj_class_target)**2)
        return no_object_loss
        
    
    
    def find_best_iou_boxes(self, box_target, box_pred):
        """
        Parameters: 
        box_target : (tensor)  size (-1, 5)
        box_pred : (tensor) size (-1, 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified 
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as
        Returns: 
        box_target_iou: (tensor)
        contains_object_response_mask : (tensor)
        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) Set the corresponding contains_object_response_mask of the bounding box with the max iou
        of the 2 bounding boxes of each grid cell to 1.
        3) For finding iou's use the compute_iou function
        4) Before using compute preprocess the bounding box coordinates in such a way that 
        if for a Box b the coordinates are represented by [x, y, w, h] then 
        x, y = x/S - 0.5*w, y/S - 0.5*h ;
        w, h = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height. 
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        5) Set the confidence of the box_target_iou of the bounding box to the maximum iou
        
        """
        
        ##### CODE #####
        box_target_iou = torch.zeros(box_target.shape)
        coo_response_mask = torch.zeros(box_target.shape)
        boxes_pred = torch.zeros(box_pred.shape[0], 4)
        boxes_target = torch.zeros(box_target.shape[0], 4)

        # x1, y1, w1, h1 = box_pred[:, 0], box_pred[:, 1], box_pred[:, 2], box_pred[:, 3]
        # x2, y2, w2, h2  = box_target[:, 0], box_target[:, 1], box_target[:, 2], box_target[:, 3]

        # box_x1, box_y1 = (x1/self.S - 0.5*w1).numpy(), (y1/self.S - 0.5*h1).numpy()
        # box_w1, box_h1 = (x1/self.S + 0.5*w1).numpy(), (y1/self.S + 0.5*h1.numpy())
        # box_x2, box_y2 = x2/self.S - 0.5*w2, y2/self.S - 0.5*h2
        # box_w2, box_h2 = x2/self.S + 0.5*w2, y2/self.S + 0.5*h2
       
        # box1_matrix = np.column_stack((box_x1, box_y1, box_w1, box_h1))
        # print(box1_matrix)
        # box1 = torch.Tensor([box_x1, box_y1, box_w1, box_h1])
        # box2 = torch.Tensor([box_x2, box_y2, box_w2, box_h2])
        # boxes_pred[i] = box1
        # boxes_target[i] = box2
        for i in range(box_target.shape[0]):
            x1, y1, w1, h1, _ = box_pred[i]
            x2, y2, w2, h2, _ = box_target[i]

            box_x1, box_y1 = x1/self.S - 0.5*w1, y1/self.S - 0.5*h1
            box_w1, box_h1 = x1/self.S + 0.5*w1, y1/self.S + 0.5*h1
            box_x2, box_y2 = x2/self.S - 0.5*w2, y2/self.S - 0.5*h2
            box_w2, box_h2 = x2/self.S + 0.5*w2, y2/self.S + 0.5*h2
            box1 = torch.Tensor([box_x1, box_y1, box_w1, box_h1])
            box2 = torch.Tensor([box_x2, box_y2, box_w2, box_h2])
            boxes_pred[i] = box1
            boxes_target[i] = box2

        iou = self.compute_iou(boxes_target, boxes_pred)

        mat = torch.diag(iou)
        a = mat.clone().detach()
        b = mat.clone().detach()
        a[::2] = a[::2] > a[1::2]
        a[1::2] = a[::2] <= a[1::2]
        a = a.bool()
        b[~a] = 0

        box_target_iou[:, 4] = b
        #take in all ones, compress to one
        coo_response_mask[a] = torch.ones(coo_response_mask.shape[1])
        #print(coo_response_mask)
        return box_target_iou, coo_response_mask.type(dtype=torch.bool)
        
    
    
    def forward(self, pred_tensor,target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30)
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes
        
        target_tensor: (tensor) size(batchsize,S,S,30)
        
        Returns:
        Total Loss
        '''
        N = pred_tensor.size()[0]
        total_loss = 0.0

        '''
        #class prediction loss
        classes_pred = pred_tensor[:,:,:,-20]
        classes_target = target_tensor[:,:,:,-20]
        class_loss = self.get_class_prediction_loss(classes_pred, classes_target)

        #regression loss
        box_pred_response = pred_tensor[:,:,:,0:10].reshape((-1,5))
        box_target_response = target_tensor[:,:,:,0:10].reshape((-1,5))
        regression_loss = self.get_regression_loss(box_pred_response, box_target_response)
        '''
        # Create 2 tensors contains_object_mask and no_object_mask 
        # of size (Batch_size, S, S) such that each value corresponds to if the confidence of having 
        # an object > 0 in the target tensor.  
        contains_object_mask = torch.zeros(pred_tensor.shape[:3], dtype=torch.bool)
        no_object_mask = torch.zeros(pred_tensor.shape, dtype=torch.bool)
        object_exists = ((target_tensor[:, :, :, 4] + target_tensor[:, :, :, 9]) > 0)
        contains_object_mask[object_exists] = True
        no_object_mask[~object_exists] = True


        # print(contains_object_mask.shape)
        ##### CODE #####

        # Create a tensor contains_object_pred that corresponds to 
        # to all the predictions which seem to confidence > 0 for having an object
        # Split this tensor into 2 tensors :
        # 1) bounding_box_pred : Contains all the Bounding box predictions of all grid cells of all images
        # 2) classes_pred : Contains all the class predictions for each grid cell of each image
        # Hint : Use contains_object_mask
        contains_object_pred = pred_tensor[contains_object_mask]
        # print(contains_object_pred.shape)
        bounding_box_pred = contains_object_pred[:,0:10].reshape((-1,5))
        # print(bounding_box_pred.shape)
        classes_pred = contains_object_pred[:,-20:]

        ##### CODE #####                   
        
        # Similarly as above create 2 tensors bounding_box_target and
        # classes_target.
        contains_object_target = target_tensor[contains_object_mask]
        # print(contains_object_target)
        bounding_box_target = contains_object_target[:,0:10].reshape((-1,5))
        classes_target = contains_object_target[:,-20:]
        
        ##### CODE #####
        # print('no obj mask', no_object_mask.shape)
        # Compute the No object loss here
        no_object_loss = self.get_no_object_loss(target_tensor, pred_tensor, no_object_mask)
        
        ##### CODE #####

        # Compute the iou's of all bounding boxes and the mask for which bounding box 
        # of 2 has the maximum iou the bounding boxes for each grid cell of each image.
        bounding_box_iou, max_iou_mask = self.find_best_iou_boxes(bounding_box_target, bounding_box_pred)
        # print(bounding_box_iou.shape)
        
        ##### CODE #####

        # Create 3 tensors :
        # 1) box_prediction_response - bounding box predictions for each grid cell which has the maximum iou
        # 2) box_target_response_iou - bounding box target ious for each grid cell which has the maximum iou
        # 3) box_target_response -  bounding box targets for each grid cell which has the maximum iou
        # Hint : Use contains_object_response_mask
        box_pred_response = bounding_box_pred[max_iou_mask].reshape((-1,5))
        # print(box_pred_response.shape)
        box_target_response_iou = bounding_box_iou[max_iou_mask].reshape((-1,5))
        # print(box_target_response_iou.shape)
        box_target_response = bounding_box_target[max_iou_mask].reshape((-1,5))
        
        ##### CODE #####
        
        # Find the class_loss, containing object loss and regression loss
        class_loss = self.get_class_prediction_loss(classes_pred, classes_target)
        contains_object_loss = self.get_contain_conf_loss(box_pred_response, box_target_response_iou)
        regression_loss = self.get_regression_loss(box_pred_response, box_target_response)
      
        ##### CODE #####
   
        total_loss = contains_object_loss +  self.l_noobj*no_object_loss + self.l_coord*regression_loss + class_loss
        return total_loss/N



