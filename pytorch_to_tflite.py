import ai_edge_torch
import numpy
import torch
import torchvision
from models.retinaface import RetinaFace
from configs.config import cfg_mnet, cfg_re50

def convert_to_tflite():
    pretrained_path = "weights/mobilenet0.25_Final.pth"
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    model = RetinaFace(cfg=cfg_mnet, phase = 'test')
    model.load_state_dict(pretrained_dict, strict=False)
    sample_inputs = torch.randn(1, 3, 640, 640)

    loc, conf, landms  = model(sample_inputs)

    edge_model = ai_edge_torch.convert(model.eval(), (sample_inputs,))

    edgeloc, edgeconf, edgelandms  = edge_model(sample_inputs)

    edge_model.export('next_RetinaFace_mobilenet0.25_640.tflite')

    print("location:",loc)
    print("conf:",conf)
    print("landms:",landms)

if __name__ == "__main__":

    convert_to_tflite()
    
