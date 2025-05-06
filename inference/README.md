# Inference Code for Masked Scene Modeling

## Setup environment

First, create a python environment and activate it:

```sh
python -m venv venv_msm_inference
source venv_msm_inference/bin/activate
```

Then, install the dependencies with the following command:

```sh
sh dependencies.sh
```

## Download pre-trained weights

You would need to download the weights of the pre-trained model. You can find them in the following [link](https://drive.google.com/file/d/1HdyQXop6nkX0_CC9NVd5Omj8_DLfYG7y/view?usp=drive_link). 


## Data format

The inference script expects a scene stored in a torch file containing a dictionary with all the relevant information. The dictionary should contain the following keys: <code>coord</code>, <code>color</code>, and <code>normal</code>.

* <code>coord</code> A numpy array of shape Nx3, with the <code>x</code>, <code>y</code>, <code>z</code> coordinates of each point. The components should be in meters and in np.float32 data type.
* <code>color</code> A numpy array of shape Nx3, with the <code>r</code>, <code>g</code>, <code>b</code> color components of each point. The components should be in the range 0-255 and in np.float32 data type. 
* <code>normal</code> A numpy array of shape Nx3, with the <code>x</code>, <code>y</code>, <code>z</code> components of the normal of each point. 


## Compute embeddings for your scene

Once the weights of the pre-trained model are downloaded and your data is in the right format, you can compute embeddings for your scene using the inference script.

```python
python inference.py -s INPUT_SCENE -w MODEL_WEIGHTS -o OUTPUT_FILE
```

If succesful, the features for each point should be stored in a torch file in the path specified in the command, <code>OUTPUT_FILE</code>.

## Integrating inference into my code

You can also compute embeddings for your scene directly from your python script. The following code provides a minimum example of how to compute the embeddings for a scene:

```python
import torch
import numpy as np
from model import build_model
from model_cfg import BACKBONE_CFG
from preprocessor import build_preprocessor

WEIGHT_PATH = ""
SCENE_PATH = ""

# Specify pre-trained weights.
BACKBONE_CFG['encoder_weight_path'] = WEIGHT_PATH

# Create the model.
msm_model = build_model(BACKBONE_CFG)
msm_model.cuda()
msm_model.eval()

# Create the scene preprocessor.
preprocessor = build_preprocessor()

# Load data. 
scene = torch.load(SCENE_PATH)
scene["normal"] = np.nan_to_num(scene["normal"])

# Pre-process data.
new_scene = preprocessor(scene)
for cur_key in new_scene.keys():
    new_scene[cur_key] = new_scene[cur_key].cuda()

# Compute embeddings.
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=True):
        out_feats = msm_model(new_scene)
```