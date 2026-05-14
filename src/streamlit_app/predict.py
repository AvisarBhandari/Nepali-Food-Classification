from torchvision import transforms
import torchvision.models as models
from PIL import Image
import torch


def pred_image(image, model, device):
    model.eval()
    weights = models.IMAGENET1K_V1.DEFAULT
    transforms = weights.transforms()
    img = Image.open(image).convert("RGB")
    img_tensor = transforms(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(img_tensor)
        prob = (torch.softmax(logits, dim=1)).squeeze()
        top_pred, top_index = torch.topk(prob, largest=True, sorted=True, k=3)
        results = []
    for pred, idx in zip(top_pred[0], top_index[0]):
        results.append({"class": pred, "Confidance": idx})
    return results


def load_model(model_path, model_fn, num_classes, device):

    checkpoint = torch.load(model_path, map_location=device)

    model = model_fn(num_classes)

    model.load_state_dict(checkpoint["model_state"])

    model.to(device)
    model.eval()

    class_names = checkpoint["class_names"]

    return model, class_names
