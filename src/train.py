import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm.auto import tqdm
from timeit import default_timer as timer
import fire
import json

from datasets import create_dataloader
from model import make_model
from engine import train_step, test_step
from torch.optim.lr_scheduler import StepLR

torch.cuda.manual_seed(42)


def train(epochs, LR, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader, class_names = create_dataloader(
        train_dir="dataset/train", test_dir="dataset/test"
    )
    model = make_model(class_names)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    start_time = timer()
    model_result = train_loop(
        MODEL_SAVE_PATH=model_path,
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
        scheduler=scheduler,
        class_name=class_names
    )
    end_time = timer()
    print("Saving Result...")
    with open("model/model_result.json", "w") as f:
        json.dump(model_result, f)
        print("model_result saved.")
    print(f"Total training time: {end_time - start_time:.3f} seconds")


def train_loop(
    MODEL_SAVE_PATH: str,
    model: torch.nn.Module,
    class_name:list,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
    epochs: int = 5,
    device: str = "cpu",
):

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    model = model.to(device)

    best_test_acc = 0
    improvement_counter = 5
    writer = SummaryWriter()

    Path("model").mkdir(exist_ok=True)
    for epoch in tqdm(range(epochs)):
        if improvement_counter >= 0:
            train_loss, train_acc = train_step(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )
            test_loss, test_acc = test_step(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )
            if scheduler:
                scheduler.step()

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                improvement_counter = 5
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "class_names": class_name,
                    },
                    MODEL_SAVE_PATH,
                )
                print(f"New Best model found ({best_test_acc}) and Saved")
            else:
                best_test_acc = test_acc

            print(
                f"Epoch: {epoch + 1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}\n patience : {improvement_counter}"
            )

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)

            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/test", test_acc, epoch)

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            improvement_counter -= 1

        else:
            print(f"traning stop as model did not improved, best Loss: {best_test_acc}")

    return results


if __name__ == "__main__":
    fire.Fire(train)
