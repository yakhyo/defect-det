import argparse
import os
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models import UNet
from utils.dataset import DamageDataset
from utils.losses import DiceCELoss, DiceLoss
from utils.misc import strip_optimizers, add_weight_decay

from utils import LOGGER

def jaccard_index(pred, target, num_classes):
    # Convert the prediction and target tensors to one-hot encoding
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    pred = torch.nn.functional.one_hot(pred, num_classes=num_classes)
    target = torch.nn.functional.one_hot(target, num_classes=num_classes)

    # Calculate the intersection and union tensors
    intersection = (pred & target).float().sum((0, 1, 2))
    union = (pred | target).float().sum((0, 1, 2))

    # Calculate the Jaccard Index for each class
    jaccard = intersection / (union + 1e-15)

    return jaccard


def train(opt, model, device):
    best_score, start_epoch = 0, 0
    best, last = f"{opt.save_dir}/best.ckpt", f"{opt.save_dir}/last.ckpt"

    # Check pretrained weights
    pretrained = opt.weights.endswith(".pt")
    if pretrained:
        ckpt = torch.load(opt.weights, map_location=device)
        model.load_state_dict(ckpt["model"].float().state_dict())
        LOGGER.info(f"Model ckpt loaded from {opt.weights}")
    model.to(device)

    # Optimizers & LR Scheduler & Mixed Precision & Loss
    parameters = add_weight_decay(model, weight_decay=opt.weight_decay)
    optimizer = torch.optim.RMSprop(parameters, lr=opt.lr, weight_decay=1e-8, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)
    criterion = DiceCELoss()

    # Resume
    if pretrained:
        if ckpt["optimizer"] is not None:
            start_epoch = ckpt["epoch"] + 1
            best_score = ckpt["best_score"]
            optimizer.load_state_dict(ckpt["optimizer"])
            LOGGER.info(f"Optimizer loaded from {opt.weights}")
            if start_epoch < opt.epochs:
                LOGGER.info(
                    f"{opt.weights} has been trained for {start_epoch} epochs. Fine-tuning for {opt.epochs} epochs"
                )
        del ckpt

    # Dataset
    trainval_dataset = DamageDataset(root="./data/train", image_size=opt.image_size, mask_suffix="")
    test_dataset = DamageDataset(root="./data/test", image_size=opt.image_size, mask_suffix="")

    # Split
    n_val = int(len(trainval_dataset) * 0.1)
    n_train = len(trainval_dataset) - n_val
    train_data, val_data = random_split(trainval_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=8, drop_last=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=8, drop_last=False, pin_memory=True)

    # Training
    for epoch in range(start_epoch, opt.epochs):
        model.train()
        epoch_loss = 0
        LOGGER.info(("\n" + "%12s" * 6) % ("Epoch", "GPU Mem", "CE Loss", "Dice Loss", "Total Loss", "iou"))
        progress_bar = tqdm(train_loader, total=len(train_loader))
        for image, target in progress_bar:
            image = image.to(device)
            target = target.to(device)

            with torch.cuda.amp.autocast(enabled=opt.amp):
                output = model(image)
                loss, losses = criterion(output, target)
                iou = jaccard_index(output, target, num_classes=7)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()
            iou = float(iou.mean())
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            progress_bar.set_description(
                ("%12s" * 2 + "%12.4g" * 4) % (f"{epoch + 1}/{opt.epochs}", mem, losses["ce"], losses["dice"], loss,iou)
            )

        dice_score, dice_loss, miou = validate(model, val_loader, device)
        LOGGER.info(f"VALIDATION: Dice Score: {dice_score:.4f}, Dice Loss: {dice_loss:.4f}, mIOU: {miou}")
        dice_score, dice_loss, miou = validate(model, test_loader, device)
        LOGGER.info(f"TEST: Dice Score: {dice_score:.4f}, Dice Loss: {dice_loss:.4f}, mIOU: {miou}")
        scheduler.step(epoch)
        ckpt = {
            "epoch": epoch,
            "best_score": best_score,
            "model": deepcopy(model).half(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(ckpt, last)
        if best_score < dice_score:
            best_score = max(best_score, dice_score)
            torch.save(ckpt, best)

    # Strip optimizers & save weights
    for f in best, last:
        new_name = os.path.splitext(f)[0] + ".pt"
        strip_optimizers(f, new_name)
        os.remove(f)


@torch.inference_mode()
def validate(model, dataloader, device, conf_threshold=0.5):
    model.eval()
    dice_score = 0
    criterion = DiceLoss()
    ious = []
    for image, target in tqdm(dataloader, total=len(dataloader)):
        image, target = image.to(device), target.to(device)
        with torch.no_grad():
            output = model(image)
            if model.out_channels == 1:
                output = F.sigmoid(output) > conf_threshold
            dice_loss = criterion(output, target)
            iou = jaccard_index(output, target, num_classes=7)
            ious.append(float(iou.mean()))
            dice_score += 1 - dice_loss
    model.train()

    return dice_score / len(dataloader), dice_loss, sum(ious)/len(ious)


def parse_opt():
    parser = argparse.ArgumentParser(description="UNet training arguments")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size, default: 512")
    parser.add_argument("--save-dir", type=str, default="weights", help="Directory to save weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs, default: 5")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size, default: 12")
    parser.add_argument("--loss", type=str, default="dice", help="Loss function, available: dice, dice_ce, focal")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate, default: 1e-5")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay, default: 1e-5")
    parser.add_argument("--weights", type=str, default="", help="Pretrained model, default: None")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--num-classes", type=int, default=7, help="Number of classes")

    return parser.parse_args()


def main(opt):
    assert opt.loss in ["dice", "dice_ce", "focal"], f"{opt.loss} not found in [`dice`, `dice_ce`, `focal`]"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Device: {device}")
    model = UNet(in_channels=3, out_channels=opt.num_classes)

    LOGGER.info(
        f"Network:\n"
        f"\t{model.in_channels} input channels\n"
        f"\t{model.out_channels} output channels (number of classes)"
    )

    # Create folder to save weights
    os.makedirs(opt.save_dir, exist_ok=True)

    train(opt, model, device)


if __name__ == "__main__":
    params = parse_opt()
    main(params)
