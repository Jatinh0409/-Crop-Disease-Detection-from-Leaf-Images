"""
gradcam.py
Grad-CAM implementation for the ResNet18 classifier: produces a heatmap
showing which regions of a leaf image drove the model's prediction.

This is the explainability layer that turns "the model says diseased" into
"the model says diseased BECAUSE of these specific spots" — which is the
part that makes the project genuinely useful, not just a classifier demo.

Usage (standalone test):
    python gradcam.py --image ../data/val/leaf_rust/leaf_rust_0000.jpg --model model.pt
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GradCAM:
    """Grad-CAM hooked onto a ResNet's final conv block (layer4)."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # global-average-pool the gradients -> per-channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx, F.softmax(output, dim=1).detach().cpu().numpy()[0]


def overlay_heatmap(original_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blends a Grad-CAM heatmap onto the original image using a simple red-hot colormap (no extra deps)."""
    original_img = original_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    orig_arr = np.array(original_img).astype(np.float32)

    heat = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    heat[..., 0] = cam * 255          # red channel scales with importance
    heat[..., 1] = (cam ** 2) * 120   # slight green for mid-range -> orange/yellow hot spots

    blended = orig_arr * (1 - alpha) + heat * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def load_model_for_gradcam(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, class_names


def predict_with_gradcam(image: Image.Image, model: torch.nn.Module, class_names: list, device: torch.device):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    cam_engine = GradCAM(model, model.layer4[-1])
    cam, class_idx, probs = cam_engine.generate(input_tensor)
    overlay = overlay_heatmap(image, cam)

    return {
        "predicted_class": class_names[class_idx],
        "confidence": float(probs[class_idx]),
        "all_class_probs": {cls: float(p) for cls, p in zip(class_names, probs)},
        "overlay_image": overlay,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="model.pt")
    parser.add_argument("--out", type=str, default="gradcam_overlay.jpg")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model_for_gradcam(args.model, device)
    image = Image.open(args.image).convert("RGB")

    result = predict_with_gradcam(image, model, class_names, device)
    result["overlay_image"].save(args.out)

    print(f"Predicted: {result['predicted_class']} (confidence={result['confidence']:.4f})")
    print(f"All class probabilities: {result['all_class_probs']}")
    print(f"Grad-CAM overlay saved to {args.out}")
