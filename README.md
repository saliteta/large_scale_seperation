# Attention Aware Seperation and Confidence Aware Filtering

## Known Issue
- During experiment with multiple GPU, some time we might have CUDA out of memory when you are using Ubuntu 20.04, currently we don't know how to solve, re-run particular block solve the issue 
- Environment set up is a little bit tricky, if one encounter that cannot install COLMAP and PYCOLMAP in original gaussian splatting, please first install COLMAP and PYCOLMAP and then install Gaussian Splatting environment

## Environment Set Up

```
conda create -n AAGS python=3.10
# Follow exactly the same as required in Gaussian Splatting
conda install -c conda-forge colmap
conda install -c conda-forge pycolmap # If it does not work, first install colmap and pycolmap and then Gaussian Splatting
```


## Method
- First Version Using heatmap work fine on the first dataset CUHK_LOWER
- Second version using simple K-means
- We currently still facing memroy efficiency problem when utilizing high resolution images


