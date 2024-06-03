import torch
from sklearn.metrics import precision_recall_fscore_support


def train_and_validate(
    model,
    device,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler,
    epochs,
    writer,
):
    """
    This function trains and validates the model and record the result in tensorboard.
    Args:
        model: The model to be trained
        device: The device to run the model on
        train_loader: The DataLoader for the training dataset
        val_loader: The DataLoader for the validation dataset
        loss_fn: The loss function to be used
        optimizer: The optimizer to be used
        scheduler: The scheduler to update the learning rate
        epochs: The number of epochs to train the model
        writer: The SummaryWriter to log the training and validation results
    """

    total_train_step = 0

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    for epoch in range(epochs):
        print(f"-----------Epoch {epoch + 1}-----------")
        model.train()

        for train_data in train_loader:
            imgs, targets = train_data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 10 == 0:
                print(f"Train step: {total_train_step}, loss: {loss.item()}")
                writer.add_scalar("Train loss", loss.item(), total_train_step)

        # Use the scheduler to update the learning rate
        scheduler.step()

        # Validation
        model.eval()

        total_val_loss = 0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for val_data in val_loader:
                imgs, targets = val_data
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_val_loss += loss.item()

                preds = outputs.argmax(1)
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)

        # Calculate precision, recall, and F1-score
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average="macro"
        )

        print(f"Average validation loss: {avg_val_loss}")
        print(f"Validation precision: {val_precision}")
        print(f"Validation recall: {val_recall}")
        print(f"Validation F1-score: {val_f1}")

        writer.add_scalar("Validation loss", avg_val_loss, epoch)
        writer.add_scalar("Validation precision", val_precision, epoch)
        writer.add_scalar("Validation recall", val_recall, epoch)
        writer.add_scalar("Validation F1-score", val_f1, epoch)

    torch.save(model.state_dict(), "model/final_model.pth")
    print("Final model saved")

    print("Training and validation completed successfully")
