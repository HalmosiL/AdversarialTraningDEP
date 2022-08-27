import torch
import numpy as np

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()

def acuracy(pred, label):
    pred = torch.nn.functional.relu(torch.sigmoid(pred))
    pred_sum = torch.sum(pred.clone(), dim=1)
    pred = torch.argmax(pred, dim=1)

    label = label.reshape(1, *label.shape)

    pred_sum = (torch.sign(pred_sum) - 1) * 200
    pred = torch.clip(pred + pred_sum, min=-1, max=20).int()

    return ((pred == label).sum() / (512 * 512)).item()
