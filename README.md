<div align="center">
  <img src="assets/gshell_logo.png" width="900"/>
</div>

# Ghost on the Shell: An Expressive Representation of General 3D Shapes


<div align="center">
  <img src="assets/teaser.png" width="900"/>
</div>

## Introduction

This is the official implementation of our paper (ICLR 2024 oral) "Ghost on the Shell: An Expressive Representation of General 3D Shapes" (G-Shell).

G-Shell is a generic and differentiable representation for both watertight and non-watertight meshes. It enables 1) efficient and robust rasterization-based multiview reconstruction and 2) template-free generation of non-watertight meshes.

Please refer to [our project page](https://gshell3d.github.io) and [our paper](https://gshell3d.github.io/static/paper/gshell.pdf) for more details.


## Getting Started

### Requirements


- Python >= 3.8
- CUDA 11.8
- PyTorch == 1.13.1

(Conda installation recommended)

#### Reconstruction

Run the following

```
pip install ninja imageio PyOpenGL glfw xatlas gdown
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
```

Follow the instructions [here](https://github.com/NVIDIAGameWorks/kaolin/) to install kaolin.

Download the tet-grid files ([res128](https://drive.google.com/file/d/1u5FzpuY_BOAg8-g9lRwvah7mbCBOfNVg/view?usp=sharing) & [res256](https://drive.google.com/file/d/1JnFoPEGcTLFJ7OHSWrI72h1H9_yOxUP6/view?usp=sharing)) to `data/tets` folder under the root directory. You shall see the folder `data/tets/` created with `256_tets.npz` inside. Alternatively, you may follow https://github.com/crawforddoran/quartet and `data/tets/generate_tets.py` to create the tet-grid files.

#### Generation

Install the following

- Pytorch3D
- ml_collections

## To-dos

- [x] Code for reconstruction
- [ ] DeepFashion3D multiview image dataset for metallic surfaces
- [ ] Code for generative models
- [ ] Code for DeepFashion3D dataset preparation
- [ ] Evaluation code for generative models

## Reconstruction

### Datasets

#### DeepFashion3D mesh dataset

We provide ground-truth images (rendered under realistic environment light with Blender) for 9 instances in [DeepFashion3D-v2 dataset](https://github.com/GAP-LAB-CUHK-SZ/deepFashion3D). The download links for the raw meshes can be found in their repo.

Training data (non-metallic material): https://drive.google.com/file/d/1LwBqLYzamFLyBIiNpD6kEkvySrq2nruG/view?usp=sharing


#### NeRF synthetic dataset

Download the [NeRF synthetic dataset archive](https://drive.google.com/uc?export=download&id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG) and unzip it into the `data/` folder.

#### Hat dataset

Download link: https://drive.google.com/file/d/18UmT1NM5wJQ-ZM-rtUXJHXkDc-ba-xVk/view?usp=sharing

### Training

#### DeepFashion3D-v2 instances

The mesh instances' IDs are [30, 92, 117, 133, 164, 320, 448, 522, 591]. To reconstruct the `$INDEX`-th mesh (in the list) using tet-based G-Shell, run

```
  python train_gshelltet_deepfashion.py --config config/deepfashion_mc_256.json --index $INDEX --trainset_path $TRAINSET_PATH --testset_path $TESTSET_PATH --o $OUTPUT_PATH
```

For FlexiCubes + G-Shell, run

```
  python train_gflexicubes_deepfashion.py --config config/deepfashion_mc_80.json --index $INDEX --trainset_path $TRAINSET_PATH --testset_path $TESTSET_PATH --o $OUTPUT_PATH
```

**NOTE: the test data are not uploaded yet. Drop `--testset_path $TESTSET_PATH` for now.**

#### Synthetic data

```
  python train_gshelltet_synthetic.py --config config/nerf_chair.json --trainset_path $TRAINSET_PATH --o $OUTPUT_PATH
```

#### Hat data

```
  python train_gshelltet_polycam.py --config config/polycam_mc_128.json --trainset_path $TRAINSET_PATH --o $OUTPUT_PATH
```

```
  python train_gshelltet_polycam.py --config config/polycam_mc_128.json --trainset_path $TRAINSET_PATH --o $OUTPUT_PATH
```

#### On config files

You may consider modify the following, depending on your demand:

- `gshell_grid`: the G-Shell grid size. For tet-based G-Shell, please make sure the corresponding tet-grid file exists under `data/tets` (e.g., `256_tets.npz`). Otherwise, follow https://github.com/crawforddoran/quartet and `data/tets/generate_tets.py` to generate the desired tet-grid file.
- `n_samples`: the number of MC samples for light rays per rasterized pixel. The higher the better (at a cost of memory and speed).
- `batch_size`: how many views sampled in each iteration.
- `iteration`: total number of iterations.
- `kd_min`, `kd_max`, etc: the min/max of the corresponding PBR material parameter.






## Generation (To be done)



## Citation

If you find our work useful to your research, please consider citing:

```
@article{liu2024gshell,
    title={Ghost on the Shell: An Expressive Representation of General 3D Shapes},
    author={Liu, Zhen and Feng, Yao and Xiu, Yuliang and Liu, Weiyang and Paull, Liam and Black, Michael J. and Sch{\"o}lkopf, Bernhard},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
}
```


## Acknowledgement

We sincerely thank the authors of [Nvdiffrecmc](https://github.com/NVlabs/nvdiffrecmc), [FlexiCubes](https://github.com/nv-tlabs/FlexiCubes) and https://github.com/yang-song/score_sde_pytorch for sharing their codes. Our repo is adapted from [MeshDiffusion](https://github.com/lzzcd001/MeshDiffusion/).