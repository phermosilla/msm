# Visualization of point embeddings

## From features to colors

First, we need to reduce our high-dimensional features to three color components. We provide a script that uses PCA to transform features into colors.

```python
python create_pca.py -s SCENE_PATH -sf SCENE_FEAT_PATH -o OUTPUT_TXT
```

The script receives three inputs:

* <code>SCENE_PATH</code> Scene in a torch file (same format as in the [inference script](/inference/README.md)). This file is used to obtain the point coordinates. 
* <code>SCENE_FEAT_PATH</code> The point features generated with the [inference script](/inference/README.md).
* <code>OUTPUT_TXT</code> Output path where the txt file will be saved. 

The output file is a text file with a line per each point in the scene. Each line contains the point coordinates and the color components separated by space: <code>x</code> <code>y</code> <code>z</code> <code>r</code> <code>g</code> <code>b</code>. The color components are in the range 0-1. You can use [Meshlab](https://www.meshlab.net/) to visualize the colored point cloud or use our blender template to have a fancy rendering with global illumination.

## Blender script

First, install [Blender](https://www.blender.org/). Then, open the provided template <code>blender_template.blend</code> using blender. Once loaded, select the <code>Scripting</code> layout **(1)** as indicated in the following image:


![Scripting](/imgs/blender_scripting.png)

If the loading script does not directly appear, use the button highlighted in the figure to load the <code>blender_load_colored_pc.py</code> script **(2)**. Then change the path to the txt file containing the coordinates and colors of each point in your scene **(3)**. The expected format is the same as the output of the <code>create_pca.py</code> script. Then, click the run script button and your point cloud should be loaded into blender and ready for rendering **(4)**. This script will first delete the existing scene <code>scene0081_00</code>. Modify the camera position and just press F12 to create the render of your scene. The result should look something similar to the following image:

![Render](/imgs/blender_render.png)