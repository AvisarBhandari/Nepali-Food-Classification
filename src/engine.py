import torch

def train_step(model:torch.nn.Module, dataloader:torch.utils.dataloader, loss_fn:torch.nn.Module,optimizer:torch.optim.Optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    for X , y in dataloader:
        X, y = X.to(device), y.to(device)
        train_logits = model(X)
        loss = loss_fn(train_logits, y)
        train_loss +=loss.item()
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(train_logits,dim=1),dim=1)
        train_acc += (y_pred_class == y).sum().item()/ len(train_logits)
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model:torch.nn.Module, dataloader:torch.utils.dataloader, loss_fn:torch.nn.Module,optimizer:torch.optim.Optimizer, device):
    model.eval()
    train_loss, train_acc = 0, 0
    with torch.inference_mode():
        for X , y in dataloader:
            X, y = X.to(device), y.to(device)
            test_logits = model(X)
            loss = loss_fn(test_logits, y)
            train_loss +=loss.item()
            y_pred_class = torch.argmax(torch.softmax(test_logits,dim=1),dim=1)
            train_acc += (y_pred_class == y).sum().item()/ len(test_logits)
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
    return train_loss, train_acc
