import torch


def collate_fn(batch):
    images, targets = zip(*batch)

    # Ensure images are tensors
    if not isinstance(images[0], torch.Tensor):
        raise TypeError(
            f"Expected images to be of type torch.Tensor but got {type(images[0])}"
        )

    images = torch.stack([img.clone().detach() for img in images], dim=0)
    targets = torch.tensor(targets)

    return images, targets
