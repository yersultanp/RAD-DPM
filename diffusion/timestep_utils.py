def map_continuous_to_teacher_timestep(t_hat, scheduler):
    # t_hat in [0,1]
    # scheduler.timesteps is a list of discrete values from teacher
    idx = int(t_hat * (len(scheduler.timesteps) - 1))
    return scheduler.timesteps[idx]
