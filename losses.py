import numpy as np
import torch
import torch.nn as nn

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, annotations, image, x_grid_order, y_grid_order, pyramid_reset, args):
        alpha = 0.25
        gamma = 2.0
        rest_norm = args.rest_norm
        s_norm = args.s_norm
        t_val = args.t_val #This value controls the Lnorm of the per pyramid loss (for a single instance)
        IOULoss = bool(args.IOU)
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        losses = []



        for j in range(batch_size):
            pyramid = pyramid_reset
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())

                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            ##build projections
            pyramid_levels = [3, 4, 5, 6, 7]
            strides = [2 ** x for x in pyramid_levels]
            image_shape = image.shape[2:]
            image_shape = np.array(image_shape)
            feature_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]

            #compute projection boxes
            projection_boxes_ls = []
            effective_boxes_ls = []
            ignoring_boxes_ls = []
            single_box_projections = torch.ones(len(pyramid_levels),5)*-1
            effective_box          = torch.ones(len(pyramid_levels),5)*-1
            ignoring_box           = torch.ones(len(pyramid_levels),5)*-1
            for single_annotation_box in bbox_annotation:
                #TODO: make sure this equation is correct
                single_box_projections[:,:4] = torch.stack([(single_annotation_box[:4] + 2 ** x - 1) // (2 ** x) for x in pyramid_levels])### NOT SURE IF THIS IS ACCURATE

                #set classes for boxes
                single_box_projections[:,4]  = single_annotation_box[4]
                effective_box[:,4]           = single_annotation_box[4]
                ignoring_box[:,4]            = single_annotation_box[4]

#                assert (single_box_projections == -1).sum() == 0, "single box projections haven't been filled with values"
                # compute coordinates of effective and ignoring and the rest, regions

                e_ef = 0.2
                e_ig = 0.5
                projections_height  = single_box_projections[:,3] - single_box_projections[:,1]
                projections_width   = single_box_projections[:,2] - single_box_projections[:,0]
                effective_box[:,3]  = single_box_projections[:,3] - ((1.0 - e_ef)/2) * projections_height
                effective_box[:,1]  = single_box_projections[:,1] + ((1.0 - e_ef)/2) * projections_height
                effective_box[:,2]  = single_box_projections[:,2] - ((1.0 - e_ef)/2) * projections_width
                effective_box[:,0]  = single_box_projections[:,0] + ((1.0 - e_ef)/2) * projections_width
                ignoring_box[:,3]   = single_box_projections[:,3] - ((1.0 - e_ig)/2) * projections_height
                ignoring_box[:,1]   = single_box_projections[:,1] + ((1.0 - e_ig)/2) * projections_height
                ignoring_box[:,2]   = single_box_projections[:,2] - ((1.0 - e_ig)/2) * projections_width
                ignoring_box[:,0]   = single_box_projections[:,0] + ((1.0 - e_ig)/2) * projections_width

#                assert (effective_box[:,3] < effective_box[:,1]).sum() == 0, "effective box not computed correctly y2 is smaller than y1"
#                assert (effective_box[:,2] < effective_box[:,0]).sum() == 0, "effective box not computed correctly x2 is smaller than x1"
#                assert (ignoring_box[:,3]  < ignoring_box[:,1]).sum()  == 0, "effective box not computed correctly y2 is smaller than y1"
#                assert (ignoring_box[:,2]  < ignoring_box[:,0]).sum()  == 0, "effective box not computed correctly x2 is smaller than x1"
                projection_boxes_ls.append(single_box_projections.tolist())
                effective_boxes_ls.append(effective_box.tolist())
                ignoring_boxes_ls.append(ignoring_box.tolist())
            #Dimensions number_of_annotation_in_image X 5_features X 4_coordinates+1_class
            projection_boxes = torch.Tensor(projection_boxes_ls)
            effective_boxes  = torch.Tensor(effective_boxes_ls)
            ignoring_boxes  = torch.Tensor(ignoring_boxes_ls)

            #Fill target maps with zeros
            #assert classification.shape == regression.shape, 'should be same shape'
            targets = torch.zeros(classification.shape)
            targets_d_left  = torch.ones(regression.shape[0]) * -1
            targets_d_right = torch.ones(regression.shape[0]) * -1
            targets_d_down  = torch.ones(regression.shape[0]) * -1
            targets_d_up    = torch.ones(regression.shape[0]) * -1
            #map instances
            instance = torch.ones(regression.shape[0]) * -1


            targets = targets.cuda()
            targets_d_left = targets_d_left.cuda()
            targets_d_right = targets_d_right.cuda()
            targets_d_down = targets_d_down.cuda()
            targets_d_up = targets_d_up.cuda()
            instance = instance.cuda()

            new_feature_idx = np.cumsum([0] + [feature_shapes[pyramid_idx][0] * feature_shapes[pyramid_idx][1] for pyramid_idx in range(len(pyramid_levels))])

            for pyramid_idx in range(len(pyramid_levels)):
                #create indices for looping over features
                next_feature_index = int(new_feature_idx[pyramid_idx + 1])
                last_feature_idx   = int(new_feature_idx[pyramid_idx])
                # TODO: There is a redundancy in this loop take care when have time
                feature_indices = torch.arange(last_feature_idx, next_feature_index).cuda().long()

                #Fill up ignoring boxes with -1
                for box in ignoring_boxes:
                    x_indices_inside_igbox = (x_grid_order[feature_indices] >= (box[pyramid_idx][0] + 1).cuda().long()) * (
                                                x_grid_order[feature_indices] <= (box[pyramid_idx][2]).cuda().long())
                    y_indices_inside_igbox = (y_grid_order[feature_indices] >= (box[pyramid_idx][1] + 1).cuda().long()) * (
                                                y_grid_order[feature_indices] <= (box[pyramid_idx][3]).cuda().long())

                    #only return indices where both x and y coordinates are inside ignoring box
                    pixel_indices_inside_igbox = x_indices_inside_igbox * y_indices_inside_igbox#THIS IS NOT AN INDEX!!!
                    #compute class
                    box_class = box[0, 4].long()
                    #from bool indices to regular indices
                    regular_pixel_indices_inside_igbox = pixel_indices_inside_igbox.nonzero().view(-1)
                    #Fill targets inside effective box with 1. (for the right class)
                    #Still have to give preference to smaller objects as in paper
                    #combine indices
                    combined_indices = feature_indices[regular_pixel_indices_inside_igbox]
                    targets[combined_indices, box_class] = -1.


                #Fill up effective boxes with 1
                for i, box in enumerate(effective_boxes):
                    x_indices_inside_effbox = (x_grid_order[feature_indices] >= (box[pyramid_idx][0]).cuda().long()) * (
                                                x_grid_order[feature_indices] <= (box[pyramid_idx][2] + 1.).cuda().long())
                    y_indices_inside_effbox = (y_grid_order[feature_indices] >= (box[pyramid_idx][1]).cuda().long()) * (
                                                y_grid_order[feature_indices] <= (box[pyramid_idx][3] + 1.).cuda().long())

                    #only return indices where both x and y coordinates are inside effective box
                    bool_pixel_indices_inside_effbox = x_indices_inside_effbox * y_indices_inside_effbox

                    #compute class
                    box_class = box[0, 4].long()
                    # from bool indices to regular indices
                    pixel_indices_inside_effbox = bool_pixel_indices_inside_effbox.nonzero().view(-1)
                    #combine indices
                    combined_indices = feature_indices[pixel_indices_inside_effbox]

                    #Fill targets inside effective box with 1. (for the right class)
                    #TODO: Still have to give preference to smaller objects in classification as in paper
                    #TODO: But maybe regression best not be trained at all?

                    targets[combined_indices, box_class] = 1.
                    projections_for_single_box = projection_boxes[i]

                    #fill regression targets inside effective box indices
                    targets_d_left[combined_indices]  = x_grid_order[combined_indices].float() - projections_for_single_box[pyramid_idx][0].cuda()
                    targets_d_right[combined_indices] = projections_for_single_box[pyramid_idx][2].cuda() - x_grid_order[combined_indices].float()
                    targets_d_down[combined_indices]  = y_grid_order[combined_indices].float() - projections_for_single_box[pyramid_idx][1].cuda()
                    targets_d_up[combined_indices]    = projections_for_single_box[pyramid_idx][3].cuda() - y_grid_order[combined_indices].float()

                    instance[combined_indices]        = i

                ######Think about adding acception for smaller objects when they overlap with larger ones....

            #compute classification loss
            alpha_factor = torch.ones(targets.shape).cuda() * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce
            #only back-propagate gradients when output not equal -1 i.e. lets set the loss to zero
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            #Normalize with number of pixels in effective region
            num_in_effective_region = (targets == 1.).sum()

            norm_sum_cls_loss = cls_loss.sum()/torch.clamp(num_in_effective_region.float(), min=1.0)
            classification_losses.append(norm_sum_cls_loss)

            # compute the loss for regression

            if num_in_effective_region == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                continue

            targets_reg = torch.stack((targets_d_left, targets_d_right, targets_d_down, targets_d_up))
            targets_reg = targets_reg.t()

            #assert ((targets_reg < -1.).sum() > 0), 'something isnt right about computing of the target regression values'

            #compute indices for effective region
            indices_for_eff_region = (targets == 1.).nonzero()[:,0]
            #only compute regression for effective region
            targets_reg = targets_reg[indices_for_eff_region]
            regression = regression[indices_for_eff_region]
            pyramid = pyramid[indices_for_eff_region]
            instance = instance[indices_for_eff_region]
            #retreive pixels from classification branch that are in the effective region and then in rest region
            class_indices_for_eff_region = (targets == 1.).nonzero()
            class_indices_for_rest_region = (targets == 0.).nonzero()
            eff_cls_loss = cls_loss[class_indices_for_eff_region[:,0],class_indices_for_eff_region[:,1]]
            rest_cls_loss = cls_loss[class_indices_for_rest_region[:,0], class_indices_for_rest_region[:,1]]

            #compute loss for each pixel
            if IOULoss:
                # normalization constant
                targets_reg = targets_reg / s_norm

                x_gt   = (targets_reg[:, 2] + targets_reg[:, 3] + 1.) * (targets_reg[:, 0] + targets_reg[:, 1] + 1.)
                x_pred = (regression[:, 2] + regression[:, 3] + 1.) * (regression[:, 0] + regression[:, 1] + 1.)

                i_h    = torch.where(targets_reg[:, 2] < regression[:, 2], targets_reg[:, 2], regression[:, 2]) + \
                         torch.where(targets_reg[:, 3] < regression[:, 3], targets_reg[:, 3], regression[:, 3]) + 1.
                i_w    = torch.where(targets_reg[:, 0] < regression[:, 0], targets_reg[:, 0], regression[:, 0]) + \
                         torch.where(targets_reg[:, 1] < regression[:, 1], targets_reg[:, 1], regression[:, 1]) + 1.
                i = i_h * i_w
                u = x_pred + x_gt - i
                IOU = i / u
                IOU = torch.clamp(IOU,1e-4, 1.0 - 1e-4)
                regression_loss = -torch.log(IOU)

            else:
                #targets_reg = targets_reg / s_norm #in anchors its 0.1
                regression_diff = torch.abs(targets_reg - regression)
                regression_loss = torch.where(torch.le(regression_diff, 1.0 / 9.0), 0.5 * 9.0 * torch.pow(regression_diff, 2), regression_diff - 0.5 / 9.0)

            #WHERE THE MAGIC HAPPENS
            losses_for_these_instances_ls = []
            follow_pyramid_losses = []

            for unique_instance in torch.unique(instance):
                losses_for_this_instance_ls = []
                for level in pyramid_levels:
                    bool_indices_for_pyramid_level = (pyramid == level)
                    bool_indices_for_instance = (instance == unique_instance)

                    combined_bool_indices = bool_indices_for_pyramid_level * bool_indices_for_instance
                    if combined_bool_indices.sum() == 0.:
                        continue
                    comb_indices = combined_bool_indices.nonzero().view(-1)
                    reg_loss_per_instance_per_pyramid_level = regression_loss[comb_indices]
                    cls_loss_per_instance_per_pyramid_level = eff_cls_loss[comb_indices]
                    assert(reg_loss_per_instance_per_pyramid_level.mean() > 0),  "weird, instance:" + str(unique_instance.item()) + " and pyramid:" + str(level) + "have regression mean zero"
                    loss_per_instance_per_pyramid_level = reg_loss_per_instance_per_pyramid_level.mean() + cls_loss_per_instance_per_pyramid_level.mean()
                    #TODO: THIS LOSS

                    losses_for_this_instance_ls.append(loss_per_instance_per_pyramid_level)


                loss_for_this_instance = torch.prod(torch.stack(losses_for_this_instance_ls))
                follow_pyramid_losses.append([round(loss.item(), 2) for loss in losses_for_this_instance_ls])
                losses_for_these_instances_ls.append(loss_for_this_instance)
            losses_for_these_instances = torch.stack(losses_for_these_instances_ls)


            total_loss = losses_for_these_instances.mean() + rest_cls_loss.mean() * rest_norm

            losses.append(total_loss)

#        return torch.stack(losses), follow_pyramid_losses
        return torch.stack(losses)
