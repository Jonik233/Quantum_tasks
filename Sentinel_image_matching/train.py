import torch
import argparse
import kornia as K
from tqdm import tqdm
import torch.optim as optim
from data_utils import load_dataloader


def train_model(epochs, dataloader, model, optimizer):
    print(f"Starting fine-tuning for {epochs} epochs...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            input_dict = {
                'image0': batch['image0'],
                'image1': batch['image1']
            }

            optimizer.zero_grad()

            outputs = model(input_dict)

            # Maximize the mean confidence of the predicted keypoints.
            if len(outputs['confidence']) > 0:
                loss = torch.mean(1.0 - outputs['confidence'])

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            else:
                progress_bar.set_postfix({'loss': "No matches"})

        print(f"Epoch {epoch + 1} Complete | Average Loss: {epoch_loss / len(dataloader):.4f}")

    save_path = "loftr_weights.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining Completed: Model weights saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dataset_dir", type=str, default="./image_matching_dataset")

    args = parser.parse_args()

    model = K.feature.LoFTR(pretrained='outdoor')

    # Freezing the CNN backbone, fine-tuning matchers only
    for param in model.backbone.parameters():
        param.requires_grad = False

    model.train()

    dataloader = load_dataloader(args.dataset_dir, args.batch_size)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    train_model(args.epochs, dataloader, model, optimizer)


if __name__ == "__main__":
    main()