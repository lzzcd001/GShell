torchrun --nnodes=1 --nproc_per_node=8 main_diffusion_ddp.py --mode=train --config=diffusion_configs/config_upper_occgrid_normalized.py \
--config.training.train_dir=$SAVE_DIR --config.data.root_dir=$REPO_ROOT_DIR
