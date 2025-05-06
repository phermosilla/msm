import torch
import argparse
import numpy as np

def get_robust_pca(features: torch.Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except:
        rins = colors
        gins = colors
        bins = colors
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])

    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def get_pca_map(
    feature_map: torch.Tensor,
    pca_stats=None,
):
    """
    feature_map: (1, h, w, C) is the feature map of a single image.
    """
    if pca_stats is None:
        reduct_mat, color_min, color_max = get_robust_pca(feature_map)
    else:
        reduct_mat, color_min, color_max = pca_stats
    pca_color = feature_map @ reduct_mat
    pca_color = (pca_color - color_min) / (color_max - color_min)
    pca_color = pca_color.clamp(0, 1)
    pca_color = pca_color.cpu().numpy()
    return pca_color


############## MAIN
if __name__ == '__main__':

    # Parse arguments.
    parser = argparse.ArgumentParser(description='Train Sematic segmentation')
    parser.add_argument('-s', default='scene.pth',
                        help='Scene file (default: scene.pth)')
    parser.add_argument('-sf', default='scene_features.pth',
                        help='Scene features file (default: scene_features.pth)')
    parser.add_argument('-o', default='scene_colors.txt',
                        help='Scene file (default: scene_colors.txt)')
    args = parser.parse_args()

    # Load coordinates.
    scene = torch.load(args.s)

    # Load features.
    scene_features = torch.load(args.sf)

    # Compute PCA.
    norm_feats = get_pca_map(scene_features.to(torch.float32))

    # Save txt
    np.savetxt(args.o,
        np.concatenate((
            scene["coord"], 
            norm_feats), -1))