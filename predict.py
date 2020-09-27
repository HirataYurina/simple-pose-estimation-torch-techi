# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:predict.py
# software: PyCharm

from lib.models.pose_resnet import get_pose_net
from lib.core.config import config
from torchvision import transforms
from lib.core.inference import get_max_preds
# from lib.core.inference import get_final_preds
import cv2
import torch
import numpy as np
import math
import colorsys
import argparse


def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='./models/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar',
                        help='the path of your model')
    parser.add_argument('--device', type=str, default='cpu',
                        help='use cpu or gpu to infer')
    parser.add_argument('--img_path', type=str, default='imgs/man5.jpg',
                        help='path of your image')
    parser.add_argument('--resize', type=tuple, default=(192, 256),
                        help='the input size of your model and the format is (width, height)')
    parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225],
                        help='the std used to normalize your images')
    parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406],
                        help='the mean used to normalize your images')
    parser.add_argument('--valid_thres', type=float, default=0.4)
    parser.add_argument('--only_up_limb', type=bool, default=True)

    parse = parser.parse_args()
    return parse


if __name__ == '__main__':

    args = argument_parse()

    pose_model = get_pose_net(config, is_train=False)

    # get weights
    weights = torch.load(args.model_path,
                         map_location=torch.device(args.device))
    # load weights
    pose_model.load_state_dict(weights)
    print('load weights successfully!!!!')
    # print(pose_model)
    # print(sum(x.numel() for x in pose_model.parameters()))

    man = cv2.imread(args.img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    img_height = man.shape[0]
    img_width = man.shape[1]
    scale_h = img_height / 256.0
    scale_w = img_width / 192.0

    man_resize = cv2.resize(man, args.resize)

    normalize = transforms.Normalize(std=args.std,
                                     mean=args.mean)

    transformation = transforms.Compose([transforms.ToTensor()])

    man_normalize = transformation(man_resize)

    # start predicting
    man_normalize = man_normalize.unsqueeze(0)
    raw_pred_res = pose_model(man_normalize)
    # (1, 17, 64, 48)
    print(raw_pred_res.shape)
    # print(raw_pred_res)

    raw_pred_res = raw_pred_res.detach().numpy()
    pred_res_192_256 = np.zeros(shape=(1, 17, img_height, img_width))
    for i in range(17):
        pred_res_192_256[0, i] = cv2.resize(raw_pred_res[0, i],
                                            (img_width, img_height),
                                            interpolation=cv2.INTER_LINEAR)

    coords, maxvals = get_max_preds(pred_res_192_256)

    heatmap_height = raw_pred_res.shape[2]
    heatmap_width = raw_pred_res.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = raw_pred_res[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                     hm[py + 1][px] - hm[py - 1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()
    print(preds)
    print(maxvals)

    maxvals = np.squeeze(maxvals)
    # TODO: optimize this threshold
    valid_thres = args.valid_thres
    valid_point = maxvals > valid_thres  # (17,)
    valid_idx = np.where(maxvals > valid_thres)
    valid_idx_list = valid_idx[0].tolist()

    preds_decode = preds
    preds_decode = np.squeeze(preds_decode)
    preds_decode = preds_decode.astype('int')

    # for point in preds_decode:
    #     cv2.circle(man,
    #                center=(int(point[0]), int(point[1])),
    #                radius=10,
    #                color=(0, 255, 0),
    #                thickness=1)

    if args.only_up_limb:
        skeleton_pair = [[12, 13], [6, 12], [7, 13], [6, 7],
                         [6, 8], [7, 9], [8, 10], [9, 11]]
    else:
        skeleton_pair = [[16, 14], [14, 12], [17, 15], [15, 13],
                         [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                         [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                         [2, 4], [3, 5], [4, 6], [5, 7]]

    skeleton = {'skeleton': skeleton_pair,
                'key_points': {0: 'nose',
                               1: 'l eye',
                               2: 'r eye',
                               3: 'l ear',
                               4: 'r ear',
                               5: 'l shoulder',
                               6: 'r shoulder',
                               7: 'l elbow',
                               8: 'r elbow',
                               9: 'l wrist',
                               10: 'r wrist',
                               11: 'l hip',
                               12: 'r hip',
                               13: 'l knee',
                               14: 'r knee',
                               15: 'l ankle',
                               16: 'r ankle'}}

    skeleton_pair = skeleton['skeleton']
    skeleton_pair = np.array(skeleton_pair, dtype='int') - 1

    # different color for every skeleton pair
    hsv_tuples = [(x / len(skeleton_pair), 1., 1.)
                  for x in range(len(skeleton_pair))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    # connect skeleton pair
    for i, pair in enumerate(skeleton_pair):
        if pair[0] in valid_idx_list and pair[1] in valid_idx_list:
            point1 = (preds_decode[pair[0]][0], preds_decode[pair[0]][1])
            point2 = (preds_decode[pair[1]][0], preds_decode[pair[1]][1])
            cv2.line(man, point1, point2, color=colors[i], thickness=3, lineType=4)

    # draw key point
    fontScale = cv2.getFontScaleFromHeight(fontFace=6,
                                           pixelHeight=img_height) / 27
    for key in valid_idx_list:
        if 4 < key < 12:
            retval, _ = cv2.getTextSize(text=skeleton['key_points'][key],
                                        fontFace=1,
                                        fontScale=fontScale,
                                        thickness=1)
            origin1 = (preds_decode[key][0] + 5, preds_decode[key][1] - 5 - retval[1])
            origin2 = (preds_decode[key][0] + 5 + retval[0], preds_decode[key][1] - 5)
            text_origin = (preds_decode[key][0] + 5, preds_decode[key][1] - 5)

            if origin2[0] > img_width:
                origin1 = (preds_decode[key][0] + 5 - retval[0], preds_decode[key][1] - 5 - retval[1])
                origin2 = (preds_decode[key][0] + 5, preds_decode[key][1] - 5)
                text_origin = (preds_decode[key][0] + 5 - retval[0], preds_decode[key][1] - 5)

            cv2.rectangle(man, origin1, origin2, (0, 0, 255), thickness=-1)

            cv2.putText(img=man,
                        text=skeleton['key_points'][key],
                        org=text_origin,
                        fontFace=1,
                        fontScale=fontScale,
                        color=(255, 255, 255),
                        thickness=1)
    # cv2.imwrite('imgs/man5_result.jpg', man)
    cv2.imshow('man', man)
    cv2.waitKey(0)
