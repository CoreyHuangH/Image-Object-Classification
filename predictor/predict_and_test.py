import torch
from sklearn.metrics import precision_recall_fscore_support


def predict_and_test(model, test_loader, loss_fn, epochs, device, writer):
    """
    This function tests a model on a test dataset and record the results in tensorboard.
    Args:
        model: The model to be tested
        test_loader: The DataLoader for the test dataset
        loss_fn: The loss function to be used
        epochs: The number of epochs to test the model
        device: The device to run the model on
        writer: The SummaryWriter to log the test results
    """

    total_test_step = 0

    for epoch in range(epochs):
        print(f"-----------Epoch {epoch + 1}-----------")
        model.eval()

        total_test_loss = 0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for test_data in test_loader:
                imgs, targets = test_data
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()

                preds = outputs.argmax(1)
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        avg_test_loss = total_test_loss / len(test_loader)

        # Calculate precision, recall, and F1-score
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average="macro"
        )

        print(f"Average test loss: {avg_test_loss}")
        print(f"Test precision: {test_precision}")
        print(f"Test recall: {test_recall}")
        print(f"Test F1-score: {test_f1}")

        writer.add_scalar("Test loss", avg_test_loss, total_test_step)
        writer.add_scalar("Test precision", test_precision, total_test_step)
        writer.add_scalar("Test recall", test_recall, total_test_step)
        writer.add_scalar("Test F1-score", test_f1, total_test_step)
        total_test_step += 1

    print("Prediction and testing completed successfully")
