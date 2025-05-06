import torch
import argparse
import numpy as np

from model import build_model
from model_cfg import BACKBONE_CFG
from preprocessor import build_preprocessor

############## MAIN
if __name__ == '__main__':

    # Parse arguments.
    parser = argparse.ArgumentParser(description='Train Sematic segmentation')
    parser.add_argument('-w', default='weights/model.pth', 
                        help='Model pre-trained weights (default: weights/model.pth)')
    parser.add_argument('-s', default='scene.pth',
                        help='Scene file (default: scene.pth)')
    parser.add_argument('-o', default='scene_features.pth',
                        help='Scene file (default: scene_features.pth)')
    args = parser.parse_args()

    # Modify backbone config.
    BACKBONE_CFG['encoder_weight_path'] = args.w

    # Create the model.
    msm_model = build_model(BACKBONE_CFG)
    msm_model.cuda()
    msm_model.eval()

    # Create the preprocessor.
    preprocessor = build_preprocessor()

    # Load data. Dictionary with the following elements: "coord", "color", "normal"
    scene = torch.load(args.s)
    scene["normal"] = np.nan_to_num(scene["normal"])

    # Transform data.
    new_scene = preprocessor(scene)
    for cur_key in new_scene.keys():
        new_scene[cur_key] = new_scene[cur_key].cuda()

    # Compute embeddings.
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            out_feats = msm_model(new_scene)

    # Save features.
    torch.save(out_feats, args.o)