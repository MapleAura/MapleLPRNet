# -*- coding: utf-8 -*-

import cv2
import numpy as np
from rknn.api import RKNN

ONNX_MODEL = '/mnt/code/13.lprnet/LPRNet/lprnet_1003.onnx'
RKNN_MODEL = '/mnt/code/13.lprnet/LPRNet/lprnet_1003.rknn'
DATASET = '/mnt/code/0.dataset/CBLPRD330K/quan.lst'

QUANTIZE_ON = True



# 注意调整为onnx模型的大小。
model_h = 24
model_w = 94


CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新', '学', '港', '澳', '警', '使', '领', '应', '急', '挂',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', '-'
        ]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    im = cv2.resize(im, new_shape, interpolation=cv2.INTER_LINEAR)
    return im

def postprocess(input_image, outputs):
    preb_labels = list()
    preb = outputs[0][0]
    preb_label = list()
    for j in range(preb.shape[1]):
        preb_label.append(np.argmax(preb[:, j], axis=0))
    no_repeat_blank_label = list()
    pre_c = preb_label[0]
    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)
    for c in preb_label: # dropout repeate label and blank label
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    preb_labels.append(no_repeat_blank_label)
    return preb_labels


    
def export_rknn():
    rknn = RKNN(verbose=True)

    rknn.config(
        mean_values=[[127.5, 127.5, 127.5]],
        std_values=[[128, 128, 128]],
        quantized_algorithm='normal',
        quantized_method='channel',
        # optimization_level=2,
        compress_weight=False,  # 压缩模型的权值，可以减小rknn模型的大小。默认值为False。
        # single_core_mode=True,
        # model_pruning=False,  # 修剪模型以减小模型大小，默认值为False。
        target_platform='rk3588'
    )
    
    output_name = [
            "out"
        ]
    
    rknn.load_onnx(
        model=ONNX_MODEL,
        outputs=output_name
    )
    rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, rknn_batch_size=1)
    rknn.export_rknn(RKNN_MODEL)

    rknn.init_runtime()
            
    return rknn





if __name__ == '__main__':
    # 数据准备
    img_path = '/mnt/code/13.lprnet/LPRNet_Pytorch/000150312.jpg'
    orig_img = cv2.imread(img_path)
    
    img = orig_img
    img_h, img_w = img.shape[:2]
    resized_img = letterbox(img, new_shape=(model_w, model_h))  # padding resize
    # resized_img = cv2.resize(img, (model_w, model_h), interpolation=cv2.INTER_LINEAR) # direct resize
    input = np.expand_dims(resized_img, axis=0)

    # 转换模型
    rknn = export_rknn()
    # 推理
    outputs = rknn.inference(inputs=[input], data_format="nhwc")
    # 后处理
    preb_labels = postprocess(resized_img, outputs)
    res = ""
    for i in preb_labels[0]:
        res += CHARS[i]
    print(res)

    # 释放
    rknn.release()
