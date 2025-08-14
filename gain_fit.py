import os
import torch
import argparse
import tqdm


from data import RandomGain
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="calculate a linear regression of gain in latent space")

parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="which device?"
)

args = parser.parse_args()


def main(args):
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(3)

    # device
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "mps":
        device = torch.device("mps")        
    else:
        device = torch.device('cpu')

    # define linear regression model
    learning_rate = 0.00001

    linear_regression = torch.nn.Linear(65, 64, bias=False).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(linear_regression.parameters(), lr=learning_rate)

    max_steps = 30000
    batch_size = 100 # idk man

    data = RandomGain('./data')
    loader = DataLoader(data, batch_size=batch_size)

    batch_rerun = 10

    steps = 0
    batches = tqdm.tqdm(loader, total=max_steps, unit="batch", desc="training")
    for inputs, labels in batches:
        if steps > max_steps:
            break
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        for _ in range(batch_rerun):
        
            outputs = linear_regression(inputs)
            loss = criterion(outputs, labels)

            # clear gradients between epochs
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            batches.set_description(f'loss: {loss.item():.5f}')
            batches.update(1)

if __name__ == "__main__":
    main(args)
