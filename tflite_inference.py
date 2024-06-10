import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from utils.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm
from utils.prior_box import PriorBox
from configs.config import cfg_mnet, cfg_re50
import torch


def load_image(image_path, input_size):
    '''
    Read image and do pre processing
    '''
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_raw = cv2.resize(img_raw, input_size)
    img = np.float32(img_raw)
    img = np.array(img)
    img = img.astype('float32')
    # img = img / 255.0
    img -= (104, 117, 123)
    img = np.expand_dims(img, axis=0)
    img = np.transpose(img, (0, 3, 1, 2))
    return img

def run_inference(model_path, image_path, input_size):
    '''
    Tflite inference of the model
    '''
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the input image
    input_data = load_image(image_path, input_size)

    # Set the tensor to point to the input data
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the inference
    interpreter.invoke()

    # Get the output
    conf = interpreter.get_tensor(output_details[0]['index'])
    loc = interpreter.get_tensor(output_details[1]['index'])
    landms = interpreter.get_tensor(output_details[2]['index'])

    return conf,loc,landms

def post_processing(model_output,image_path,input_size,i):
    '''
    Post processing to convert the raw prediction to the pixel values
    '''
    confidence_threshold = 0.02
    top_k = 500
    keep_top_k = 100
    nms_threshold = 0.4
    vis_thres = 0.9
    save_image = True
    conf,loc,landms = model_output
    variance = [0.1,0.2]
    cfg = cfg_mnet
    
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_raw = cv2.resize(img_raw, input_size)
    img = np.array(img_raw)
    resize = 1
    im_height, im_width, _ = img.shape
    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    prior_data = priors.data

    boxes =  np.squeeze(loc)
    boxes = decode(torch.from_numpy(boxes), prior_data, cfg['variance'])

    boxes = boxes * scale / resize
    # boxes = boxes.cpu().numpy()
    scores =  np.squeeze(conf)[:, 1]
    landms = np.squeeze(landms)
    landms = decode_landm(torch.from_numpy(landms), prior_data, variance)

    scale1 = np.array([im_height, im_width, im_height, im_width,im_height, im_width,im_height, im_width,im_height, im_width])
    # scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    # landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = np.copy(scores.argsort()[::-1][:top_k])
    # print(order)
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    # show image
    if save_image:
        for b in dets:
            if b[4] < vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        # save image

        name = "data/data_faces_pred/testtflite"+str(i)+".jpg"
        cv2.imwrite(name, img_raw)

if __name__ == '__main__':
    import os
    from glob import glob
    # Define the paths
    model_path = 'RetinaFace_mobilenet0.25_640.tflite'

    # Define the input size (height, width)
    input_size = (640, 640)
    image_paths = glob(os.path.join("data/data_faces","*"))
    for i,image_path in enumerate(image_paths):
        # Run inference
        output1,output2,output3 = run_inference(model_path, image_path, input_size)
        post_processing((output1,output2,output3),image_path,input_size,i)


