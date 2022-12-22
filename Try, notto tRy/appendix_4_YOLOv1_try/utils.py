import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):
    if box_format == 'midpoint':
        box1_x1 = boxes_preds[...,0:1] - boxes_preds[...,2:3]/2
        box1_y1 = boxes_preds[...,1:2] - boxes_preds[...,3:4]/2
        box1_x2 = boxes_preds[...,0:1] + boxes_preds[...,2:3]/2
        box1_y2 = boxes_preds[...,1:2] + boxes_preds[...,3:4]/2
        box2_x1 = boxes_labels[...,0:1] - boxes_labels[...,2:3]/2
        box2_y1 = boxes_labels[...,1:2] - boxes_labels[...,3:4]/2
        box2_x2 = boxes_labels[...,0:1] + boxes_labels[...,2:3]/2
        box2_y2 = boxes_labels[...,1:2] + boxes_labels[...,3:4]/2

    if box_format == 'corners':
        box1_x1 = boxes_preds[...,0:1]
        box1_y1 = boxes_preds[...,1:2]
        box1_x2 = boxes_preds[...,2:3]
        box1_y2 = boxes_preds[...,3:4] #n,1
        box2_x1 = boxes_preds[...,0:1]
        box2_y1 = boxes_preds[...,1:2]
        box2_x2 = boxes_preds[...,2:3]
        box2_y2 = boxes_preds[...,3:4] #n,1

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    #when teht do not intersect
    intersection = (x2-x1).clamp(0)*(y2-y1).clamp(0)

    box1_area = abs((box1_x2- box1_x1)*(box1_y2-box1_y1))
    box2_area = abs((box2_x2- box2_x1)*(box2_y2-box2_y1))

    return intersection/(box1_area+box2_area - intersection+1e-6)

def non_max_suppression(bboxes, iou_threshold, threshold, box_format='corners'):
    assert type(bboxes) == []
    
    bboxes = [box for box in bboxes if box[1] > threshold] #box = predicted class, best confidence, converted bboxes
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop()

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:], torch.tensor(box[2:]),
                box_format=box_format)
            )< iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def mean_average_precision(
    pred_boxes, treu_boxes, iou_threshold=.5, box_format ='midpoint', num_classes=20
):
#list storing all ap for respective classes
    average_precisions = []

    #used ofr numerical stabilitty later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        #go thregh all prediction s and target,
        #and only add the ones that belong to the current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in treu_boxes:
            if true_box[1]==c:
                ground_truths.append(true_box)

        # find the amount of bboxes for eatch trainin ex
        #counter here findd how mayny groun truth bboxes we get
        #for each trainig example, so lets say img 0 has 3
        #img 1 has 5 then we weill obtain a ditionary with:
        #amount bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        #we then go through each key, val in this dictionary
        #and convert to the following w.r.t same
        #amount bboxes = {0: torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        #sort by box prebabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        #if noone exist ofr this class the nwe can sfaly skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detection):
            #only take out the groun treh that have the same trainin idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0]==detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                #only detect groun treuth dtetion once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    #true positive and add this boungding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            #if iou is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes+epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        #torch.trapz for numerical intergration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions)/ len(average_precisions)

def plot_image(image, boxes):
    #plot predcit boudn bbxos in image
    im = np.array(image)
    height, width,_ = im.shape

    #create figure and zxes
    fig, ax = plt.subplots(1)
    #display the image
    ax.imshow(im)

    #box0 is x mindpoint , box2 is width
    #box1 is y midpoint box3 is height

    #create a rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4 #got more values than in x,y, w, h, in a box
        upper_left_x = box[0] - box[2]/2
        upper_left_y = box[1] - box[3]/2
        rect = patches.Rectangle(
            (upper_left_x*width, upper_left_y*height),
            box[2]*width,
            box[3]*height,
            linewidth = 1,
            edgecolor = 'r',
            facecolor = 'none'
        )
        #add the patch oto the axes
        ax.add_patch(rect)

    plt.show()

def get_bboxes(
    loader, 
    model,
    iou_threshold,
    threshold,
    pred_format='cells',
    box_format = 'midpoint',
    device='cuda'
):
    all_pred_boxes = []
    all_true_boxes = []

    #make sure model is in eval bvrefor get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader): #x = image(3, 448,448) labels  = (7,7,30) batch = 16(*3CH)
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
            bboxes[idx],
            iou_threshold=iou_threshold,
            threshold = threshold,
            box_format = box_format)

            #if batch idx==9 and idx ===0:
            
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx]+nms_box)

            for box in true_bboxes[idx]:
                #many wiil get conveted to 0 pred
                if box[1]>threshold:
                    all_true_boxes.append([train_idx]+box) #all treu booxes shape = 16,6 6 = class, conficenc cx,cy,w,,h

            train_idx +=1

    model.train() #update batchnorm mean, std
    return all_pred_boxes, all_true_boxes

def convert_cellboxes(predictions, S=7):
    predictions= predictions.to('cpu')
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7,7,30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26,30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[...,25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1*(1-best_box) + best_box*bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7,1).unsqueeze(-1)
    x = 1/S*(best_boxes[...,:1]+cell_indices)
    y = 1/S*(best_boxes[...,1:2]+cell_indices.permute(0,2,1,3))
    w_y = 1/S*best_boxes[...,2:4]
    converted_bboxes = torch.cat((x,y,w_y), dim=-1)
    predicted_class = predictions[...,:20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[...,20], predictions[...,25]).unsqueeze(-1)
    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)

    return converted_preds

def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S*S,-1) #16,7*7, 6
    converted_pred[...,0] = converted_pred[...,0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]): #batch 16
        bboxes = []

        for bbox_idx in range(S*S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename='my_checpointpth.tar'):
    print('svaeiong checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print('loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


