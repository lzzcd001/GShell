import torch
import os
import logging


def restore_checkpoint(ckpt_dir, state, device, strict=False, rank=None):
  if not os.path.exists(ckpt_dir):
    os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    if strict:
      raise
    return state
  else:
    if rank is not None:
      device = f"cuda:{rank}"
    # loaded_state = torch.load(ckpt_dir, map_location=device)
    loaded_state = torch.load(ckpt_dir, map_location='cpu')
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    try:
      state['model'].load_state_dict(loaded_state['model'], strict=False)
    except:
      consume_prefix_in_state_dict_if_present(loaded_state['model'])
      state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'], device=device)
    state['step'] = loaded_state['step']
    state['model'].to(device)
    try:
      state['gradscaler'].load_state_dict(loaded_state['gradscaler'])
      # state['gradscaler'].to(device)
    except:
      # raise
      pass
    torch.cuda.empty_cache()
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step'],
    'gradscaler': state['gradscaler'].state_dict()
  }
  torch.save(saved_state, ckpt_dir)