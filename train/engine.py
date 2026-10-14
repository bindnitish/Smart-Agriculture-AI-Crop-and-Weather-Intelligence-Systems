import torch


# ==========================================
# TRAIN ONE EPOCH
# ==========================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        # Move data to GPU/CPU
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Loss
        running_loss += (
            loss.item() * images.size(0)
        )

        # Predictions
        _, predicted = torch.max(
            outputs,
            dim=1
        )

        # Accuracy
        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = running_loss / total

    epoch_accuracy = (
        correct / total
    ) * 100

    return epoch_loss, epoch_accuracy


# ==========================================
# VALIDATION
# ==========================================

def validate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # No gradients during validation
    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Loss
            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            # Predictions
            _, predicted = torch.max(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    epoch_loss = running_loss / total

    epoch_accuracy = (
        correct / total
    ) * 100

    return epoch_loss, epoch_accuracy