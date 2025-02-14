import torch
from einops import rearrange
from frogbox.utils import load_model_checkpoint
from torchdiffeq import odeint

from datasets.rooms import Rooms
from utils.save import save_image

CKPT_PATH = ""
DEVICE = "cuda:2"
MAX_EDGE = None # None for training resolution
STEPS = 16
N_SAMPLES = 10

OUTPUT = [
    "TRAJECTORY",
    "IMAGE",
]


def main():

    device = torch.device(DEVICE)
    model, config = load_model_checkpoint(CKPT_PATH)
    model = model.eval().to(device)
    
    num_latents = config.model.params['num_latents']
    latent_dim = config.model.params['latent_dim']
    
    dataset = Rooms(**config.datasets["val"].params) # only 
    max_edge = dataset.max_edge if MAX_EDGE is None else MAX_EDGE

    for idx in range(N_SAMPLES):

        # Prepare noise and timesteps
        noise = torch.randn(1, 3, max_edge, max_edge, device=device)
        prev_latent = torch.zeros(1, num_latents, latent_dim, device=device)
        timesteps = torch.linspace(0, 1, STEPS, device=device)

        # Integrate the model
        def ode_func(t, y):
            image, prev_latent = y
            t = t.reshape(-1,)
            prev_latent, flow = model((image, t, prev_latent))
            return (flow, torch.zeros_like(prev_latent)) # no flow for the latent

        with torch.inference_mode():
            trajectory = odeint(ode_func, (noise, prev_latent), timesteps, method="midpoint", atol=1e-5, rtol=1e-5)
        trajectory = trajectory[0].cpu().detach().squeeze(1) # drop latents
        
        trajectory = trajectory.clamp(0, 1)
        image = trajectory[-1]

        # Output
        name = str(idx).zfill(4)
        if "IMAGE" in OUTPUT:
            save_image(image, f"data/output/{name}_image.png")
        if "TRAJECTORY" in OUTPUT:
            trajectory = rearrange(trajectory, "s c h w -> s c h w")
            trajectory = torch.cat([sample for sample in trajectory], dim=-1)
            save_image(trajectory, f"data/output/{name}_trajectory.png")

        print(f"Saved {name}")


if __name__ == "__main__":
    main()