import ml_collections
import torch
import os


def get_config():
    config = ml_collections.ConfigDict()

    # data
    data = config.data = ml_collections.ConfigDict()
    data.root_dir = 'PLACEHOLDER'
    # data.dataset_metapath = os.path.join(data.root_dir, 'metadata/lower_res64_train.txt')
    data.num_workers = 4
    data.grid_size = 128
    data.tet_resolution = 64
    data.num_channels = 4
    data.use_occ_grid = True
    data.grid_metafile = os.path.join(data.root_dir, 'metadata/lower_res64_grid_train.txt')
    data.occgrid_metafile = os.path.join(data.root_dir, 'metadata/lower_res64_occgrid_train.txt')

    data.occ_mask_path = os.path.join(data.root_dir, 'data/occ_mask_res64.pt')
    data.tet_info_path = os.path.join(data.root_dir, 'data/tet_info.pt')

    data.filter_meta_path = None
    data.aug = True

    # training
    training = config.training = ml_collections.ConfigDict()
    training.sde = 'vpsde'
    training.continuous = False
    training.reduce_mean = True
    training.batch_size = 1 ### for DDP, global_batch_size = nproc * local_batch_size
    training.num_grad_acc_steps = 4 
    training.n_iters = 2400001
    training.snapshot_freq = 1000
    training.log_freq = 50
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.loss_type = 'l2'
    training.train_dir = "PLACEHOLDER"
    training.snapshot_freq_for_preemption = 1000
    training.gradscaler_growth_interval = 1000
    training.use_aux_loss = False


    training.compile = True # PyTorch 2.0, torch.compile
    training.enable_xformers_memory_efficient_attention = True

    # sampling
    sampling = config.sampling = ml_collections.ConfigDict()
    sampling.method = 'pc'
    sampling.predictor = 'ancestral_sampling'
    sampling.corrector = 'none'
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.075


    # model
    model = config.model = ml_collections.ConfigDict()
    model.name = 'unet3d_occgrid'
    model.use_occ_grid = True
    model.num_res_blocks = 2
    model.num_res_blocks_1st_layer = 2
    model.base_channels = 128
    model.ch_mult = (1, 2, 2, 4, 4, 4)
    model.down_block_types = (
        "ResBlock", "ResBlock", "ResBlock", "AttnResBlock", "ResBlock", "ResBlock"
    )
    model.up_block_types = (
       "ResBlock", "ResBlock", "AttnResBlock", "ResBlock", "ResBlock", "ResBlock"
    )
    model.scale_by_sigma = False
    model.num_scales = 1000
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.act_fn = 'swish'
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.dropout = 0.1
    model.sigma_max = 378
    model.sigma_min = 0.01
    model.beta_min = 0.1
    model.beta_max = 20.
    model.embedding_type = 'fourier'
    model.pred_type = 'noise'
    model.conditional = True

    model.feature_mask_path = os.path.join(data.root_dir, 'data/global_mask_res64.pt')
    model.pixcat_mask_path = os.path.join(data.root_dir, 'data/cat_mask_res64.pt')

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 1e-5
    optim.optimizer = 'AdamW'
    optim.lr = 1e-5
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    # eval
    config.eval = eval_config = ml_collections.ConfigDict()
    eval_config.batch_size = 2
    eval_config.idx = 0
    eval_config.bin_size = 30
    eval_config.eval_dir = "PLACEHOLDER"
    eval_config.ckpt_path = "PLACEHOLDER"
    

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


    return config
