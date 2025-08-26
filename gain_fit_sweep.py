import os
import torch
import argparse
import tqdm
import wandb
import random
import itertools
from pathlib import Path

from torch.utils.data import DataLoader, random_split
import torchaudio

from data import RandomGain
from models import Effect
from utils import get_audio_files

"""
DONE: validate takes average
DONE: checkpoint every n steps (1000?)
TODO: examples
"""


def save_model(
    model, step, optimizer, scheduler, loss, run_id, save_dir="./checkpoints"
):
    """Saves the model checkpoint in a run-specific directory."""
    run_dir = Path(save_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / f"checkpoint_step_{step}.pt"

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
            if scheduler
            else None,  # Save scheduler state
            "loss": loss,
        },
        save_path,
    )
    return save_path


@torch.no_grad()
def make_examples(loader, model, m2l, device):
    # need to:
    # - TODO: load up 10 second snippets from random audio files
    # - TODO: encode them to latent
    # - TODO: add gain val
    # - TODO: run through model
    # - TODO: decode
    # - TODO: post to wandb

    # get random audio file
    audio_file = loader.files[random.randint(0, len(loader.files) - 1)]

    # load and resmaple
    audio, sr = torchaudio.load(audio_file)

    # only take first channel
    if audio.shape[0] > 1:
        audio = audio[0]

    resampler = torchaudio.transforms.Resample(sr, loader.sr)
    audio = resampler(audio)

    # get random 10s snippet
    pos = random.randint(0, audio.shape[0])
    audio = audio[pos : pos + 10 * loader.sr]

    # get random gain
    gain = random.random()

    # encode
    latent = m2l.encode(audio).squeeze().to(device)
    gain_tensor = torch.full((1, latent.shape[-1]), gain).to(device)
    latent = torch.cat([latent, gain_tensor], 0)
    latent = latent.permute(1, 0)

    latent = model(latent)
    latent = latent.permute(1, 0)
    new_audio = m2l.decode(latent)

    return {
        "val/orig": wandb.Audio(audio, sample_rate=loader.sr),
        "val/gain": wandb.Audio(new_audio[0], sample_rate=loader.sr),
    }


def run_training():
    """
    training and validation process.
    """
    with wandb.init() as run:
        config = wandb.config

        # --- Setup ---
        # Set seeds for reproducibility
        torch.manual_seed(config.seed)
        torch.set_float32_matmul_precision("high")

        # Set up device
        if config.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif config.device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"INFO: Using device: {device}")

        # --- Data Loading and Splitting ---

        # train/validation split
        files = get_audio_files(config.data_dir)
        random.shuffle(files)

        val_size = 1000

        val_files = files[:val_size]
        train_files = files[val_size:]

        train_dataset = RandomGain(config.data_dir, device=device, files=train_files)
        val_dataset = RandomGain(config.data_dir, device=device, files=val_files)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
        # val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        val_loader = train_loader
        print(
            f"INFO: Training on {len(train_files)} samples, validating on {len(val_files)} samples."
        )

        # --- Model, Loss, and Optimizer ---
        if config.model_type == "effect":
            model = Effect(
                65,
                64,
                hidden_dims_scale=config.hidden_dims_scale,
                num_inner_layers=config.num_inner_layers,
                resid=False,
            ).to(device)
        elif config.model_type == "linear":
            model = torch.nn.Linear(65, 64, bias=False).to(device)
        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")

        # Log model architecture and gradients with wandb
        # wandb.watch(model, log="all", log_freq=100)

        criterion = torch.nn.MSELoss()

        if config.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        # --- Training & Validation Loop ---
        steps = 0
        best_val_loss = float("inf")
        pbar = tqdm.tqdm(total=config.max_steps, unit="step", desc="Training")

        avg_val_loss = -1
        while steps < config.max_steps:
            for inputs, labels in train_loader:
                if steps >= config.max_steps:
                    break

                # if steps >= 499:
                #     breakpoint()

                model.train()
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Log training metrics
                training_log = {}
                wandb.log({"train_loss": loss.item()}, step=steps)

                steps += 1
                pbar.update(1)
                pbar.set_postfix(
                    train_loss=f"{loss.item():.5f}", val_loss=f"{avg_val_loss:.5f}"
                )

                # --- Periodic Validation ---
                if steps % config.val_interval == 0 or steps == 1:
                    model.eval()

                    examples = make_examples(
                        val_dataset, model, val_dataset.encdec, device
                    )
                    val_loss = 0.0
                    iter_count = 100
                    with torch.no_grad():
                        for val_inputs, val_labels in itertools.islice(
                            val_loader, iter_count
                        ):
                            val_outputs = model(val_inputs.to(device))
                            val_loss += criterion(
                                val_outputs, val_labels.to(device)
                            ).item()

                    avg_val_loss = val_loss / iter_count
                    wandb.log({"val_loss": avg_val_loss} | examples, step=steps)

                    # Save the model if it's the best so far
                    # if avg_val_loss < best_val_loss:
                    #     best_val_loss = avg_val_loss
                    #     wandb.summary['best_val_loss'] = best_val_loss
                    #     model_path = os.path.join(wandb.run.dir, "best_model.pth")
                    #     torch.save(model.state_dict(), model_path)

                if steps % config.checkpoint_every == 0 and steps > 0:
                    checkpoint_path = save_model(
                        model,
                        steps,
                        optimizer,
                        None,
                        loss,
                        run.id,
                        "/scratch/nshaheed/gain_fit",
                    )
                    # Log checkpoint as a versioned artifact
                    artifact = wandb.Artifact(f"model-ckpt-{run.id}", type="model")
                    artifact.add_file(local_path=checkpoint_path, name="model.pt")
                    run.log_artifact(artifact)

        pbar.close()
        print("INFO: Training finished.")
        # Save the final model artifact to wandb
        # final_model_path = os.path.join(wandb.run.dir, "final_model.pth")
        # torch.save(model.state_dict(), final_model_path)


if __name__ == "__main__":
    # The parser allows running the script for a single, non-sweep run.
    # The wandb agent will ignore these and use the sweep config.
    # parser = argparse.ArgumentParser(description="Hyperparameter Sweep for music2latent gain model")
    # parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    # args = parser.parse_args()

    # Default configuration for a single run.
    # These values will be overridden by the sweep agent.
    config_defaults = {
        "device": "cuda",
        "data_dir": "/scratch/nshaheed/mtg-jamendo/",
        "seed": 42,
        "max_steps": 10000,
        "val_interval": 500,  # How often to run validation (in steps)
        "learning_rate": 1e-3,
        "batch_size": 256,
        "hidden_dims_scale": 6,
        "num_inner_layers": 6,
        "optimizer": "adam",
        "model_type": "effect",
    }

    # Initialize W&B, tell it about the project and the default config
    wandb.init(config=config_defaults, project="music2latent-gain")

    # sweeping
    # Define a sweep config dictionary
    sweep_configuration = {
        "method": "bayes",
        "name": "fifth-sweep",
        "metric": {"goal": "minimize", "name": "val_loss"},
        "parameters": {
            "data_dir": {"value": "/scratch/nshaheed/mtg-jamendo/"},
            "device": {"value": "cuda"},
            "seed": {"value": 42},
            "val_interval": {"value": 500},
            "optimizer": {"value": "adam"},
            "model_type": {"value": "effect"},
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1.2e-4,
                "max": 2.5e-4,
            },
            "batch_size": {"value": 256},
            "max_steps": {"value": 80001},
            "hidden_dims_scale": {"values": [6, 8, 10]},
            "num_inner_layers": {"value": 1},
            "checkpoint_every": {"value": 1000},
        },
    }

    # run_training

    # Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
    wandb.agent(sweep_id, function=run_training, count=3)
