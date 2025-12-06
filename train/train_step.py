# train/train_step.py

def train_step(batch, scheduler_model, teacher_models, optimizer, sched_config):
    teacher_latent, teacher_image, prompt = batch

    # TODO: encode prompt → cond
    # run K-step student sampler
    # compute loss
    # backprop
    pass
