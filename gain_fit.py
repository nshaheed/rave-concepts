import os
import torch
import argparse
import tqdm


from data import RandomGain
from models import Effect
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(
    description="calculate a linear regression of gain in latent space"
)

parser.add_argument("--device", type=str, default="cpu", help="which device?")

parser.add_argument("--data", type=str, help="data dir")

args = parser.parse_args()


def main(args):
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(3)

    # device
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # define linear regression model
    learning_rate = 0.000001

    model = None
    if False:
        model = torch.nn.Linear(65, 64, bias=False).to(device)
    else:
        model = Effect(65, 64, resid=False).to(device)

    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    max_steps = 1000000
    batch_size = 256  # idk man
    # batch_size = 32 # idk man

    data = RandomGain(args.data, device)
    loader = DataLoader(data, batch_size=batch_size)

    batch_rerun = 1

    steps = 0
    batches = tqdm.tqdm(loader, total=max_steps, unit="batch", desc="training")
    for inputs, labels in batches:
        if steps > max_steps:
            break

        inputs = inputs.to(device)
        labels = labels.to(device)

        loss_avg = 0

        for _ in range(batch_rerun):
            model.train(True)

            # if (steps % 200 == 0):
            #     breakpoint()

            # clear gradients between epochs
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            loss_avg += loss.item()
            batches.update(1)
            steps += 1

        loss_avg = loss_avg / batch_rerun
        batches.set_description(f"loss: {loss_avg:.5f}")


if __name__ == "__main__":
    main(args)
