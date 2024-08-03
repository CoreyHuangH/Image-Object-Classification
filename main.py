import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50
from dataset.coco import COCODataset
from trainer.train_and_validate import train_and_validate
from predictor.predict_and_test import predict_and_test
from transforms.transforms import *
from utils import collate_fn


def main():
    root = "./my_coco_subset"  # Path to the COCO subset dataset
    targets = [
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
    ]  # The categories to be classified
    target_map = {
        target: idx for idx, target in enumerate(targets)
    }  # Map targets to numeric types

    # Create datasets for each category
    datasets = [
        COCODataset(root, target, transform=initial_transform()) for target in targets
    ]

    # Assign numeric targets
    for dataset, target in zip(datasets, targets):
        dataset.target = target_map[target]

    # Combine all datasets
    combined_dataset = ConcatDataset(datasets)

    # Split the dataset into train, val, test sets
    data_len = len(combined_dataset)
    train_size = int(0.8 * data_len)
    val_size = int(0.1 * data_len)
    test_size = data_len - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size, test_size]
    )

    # Adjust transforms
    train_set.transform = train_transform()
    val_set.transform = val_transform()
    test_set.transform = test_transform()

    # Create dataloaders
    train_loader = DataLoader(
        train_set, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    # Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and modify the last layer
    model = resnet50(weights="DEFAULT")
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        # Add dropout layer with 50% probability
        nn.Dropout(0.5),
        # Add a linear layer in order to deal with 5 classes
        nn.Linear(num_ftrs, len(targets)),
    )
    # print(model)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    learning_rate = 1e-5
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-3
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    epochs = 350

    writer = SummaryWriter("logs")

    # Train and cross-validate
    train_and_validate(
        model,
        device,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        scheduler,
        epochs,
        writer,
    )

    # Predict and test
    predict_and_test(model, test_loader, loss_fn, epochs, device, writer)

    writer.close()


if __name__ == "__main__":
    main()
