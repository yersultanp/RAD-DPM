from diffusers import DPMSolverMultistepScheduler

def compare_fixed_vs_learned(learned_schedule, K_STEPS, pipe, prompt):

    # Initialize DPM
    dpm = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Ask for the "Natural" 4 steps
    dpm.set_timesteps(num_inference_steps=K_STEPS)

    # Get the exact values
    natural_steps = dpm.timesteps.tolist()
    print(f"DPM Natural Steps: {natural_steps}")
    print(f"Learned Steps: {learned_schedule}")

    # Force DPM to use YOUR steps
    dpm.set_timesteps(timesteps=learned_schedule)

    # Now when you run the pipeline, it uses DPM's math (2nd order integration)
    # but YOUR timeline.
    image = pipe(prompt, num_inference_steps=K_STEPS).images[0]
