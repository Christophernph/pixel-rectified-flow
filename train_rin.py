import argparse
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, cast

import torch
from frogbox import SupervisedConfig, SupervisedPipeline, read_json_config
from frogbox.config import parse_log_interval


def sample_lognorm(
    size: Sequence[int],
    generator: torch.Generator = None,
    m: float = 0.0,
    s: float = 1.0,
) -> torch.FloatTensor:
    device = generator.device if generator else None
    x = m + s * torch.randn(size=size, generator=generator, device=device)
    return 1.0 / (1.0 + torch.exp(-x))

def parse_arguments(
    args: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=Path, default="configs/example.json"
    )
    parser.add_argument(
        "-d", "--device", type=torch.device, default=torch.device("cuda:0")
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-keys", type=str, nargs="+")
    parser.add_argument(
        "--logging",
        type=str,
        choices=["online", "offline"],
        default="online",
    )
    parser.add_argument("--wandb-id", type=str, required=False)
    parser.add_argument("--tags", type=str, nargs="+")
    parser.add_argument("--group", type=str)
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments()
    config = cast(SupervisedConfig, read_json_config(args.config))
    device = torch.device(args.device)
    log_interval = parse_log_interval(config.log_interval)
    self_condition_rate = config.self_condition_rate

    schedule = partial(sample_lognorm, m=0.0, s=1.0)
    train_generator = torch.Generator(device).manual_seed(1234)
    val_generator = torch.Generator(device)

    def input_transform(x, y, generator):       
        # x is ignored, as we generate the noise here
        image = y
        bs = image.size(0)
        
        noise = torch.empty_like(image).normal_(generator=generator)
        timesteps = schedule((bs,), generator).to(device, torch.float32)

        padded_times = timesteps.reshape(-1, 1, 1, 1)
        noised = padded_times * image + (1.0 - padded_times) * noise # 0 = noise, 1 = image
        flow = image - noise
        
        # The in-equality is reversed here intentionally
        # The model computes the prev_latent from zero-latent if the condition is not met
        # If zero-latent is passed, the model will use that as the prev_latent
        prev_latents = None
        if torch.rand(1, device=image.device, generator=generator) > self_condition_rate:
            prev_latents = torch.zeros(bs,
                                       config.model.params['num_latents'],
                                       config.model.params['latent_dim'],
                                       device=device)

        return (noised, timesteps, prev_latents), flow

    def model_transform(output):
        return output[1] # only return the flow
    
    def reset_val_rng(pipeline: SupervisedPipeline):
        pipeline.datasets["val"].reset_rng()
        val_generator.manual_seed(5678)

    pipeline = SupervisedPipeline(
        config=config,
        device=args.device,
        checkpoint=args.checkpoint,
        checkpoint_keys=args.checkpoint_keys,
        logging=args.logging,
        wandb_id=args.wandb_id,
        tags=args.tags,
        group=args.group,
        trainer_input_transform=partial(
            input_transform, generator=train_generator
        ),
        evaluator_input_transform=partial(
            input_transform, generator=val_generator
        ),
        trainer_model_transform=model_transform,
        evaluator_model_transform=model_transform,
    )

    model_forward = pipeline.model.forward
    # pipeline.model.forward = lambda inputs: model_forward(**inputs)

    pipeline.install_callback(log_interval, reset_val_rng)

    num_params = sum(p.numel() for p in pipeline.model.parameters())
    trainable_params = sum(p.numel() for p in pipeline.model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M, trainable: {trainable_params / 1e6:.2f}M")

    pipeline.run()
