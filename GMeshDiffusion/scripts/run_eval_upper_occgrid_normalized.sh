python main_diffusion.py --mode uncond_gen --config diffusion_configs/config_upper_occgrid_normalized.py \
--config.eval.eval_dir=$EVAL_DIR \
--config.data.root_dir=$REPO_ROOT_DIR \
--config.sampling.method=ddim \
--config.eval.ckpt_path=$CKPT_PATH \
--config.eval.bin_size=10 \
--config.eval.idx $1