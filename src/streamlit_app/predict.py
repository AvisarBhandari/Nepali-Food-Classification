from torchvision import transforms
import torchvision.models as models
from PIL import Image
import torch

IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
def pred_image(image, model,class_names, device):
    model.eval()
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(img_tensor)
        prob = (torch.softmax(logits, dim=1)).squeeze()
        top_pred, top_index = torch.topk(prob, largest=True, sorted=True, k=3)
        results = []
        # top_pred = top_pred.squeeze(0)
        # top_index = top_index.squeeze(0)
    for pred, idx in zip(top_pred, top_index):
        results.append({"class": class_names[idx.item()], "Confidance": f"{pred.item() * 100:.2f}"})
    return results


def load_model(model_path, model_fn, num_classes, device):

    checkpoint = torch.load(model_path, map_location=device)

    model = model_fn(num_classes)

    model.load_state_dict(checkpoint["model_state"])

    model.to(device)
    model.eval()

    class_names = checkpoint["class_names"]

    return model, class_names
