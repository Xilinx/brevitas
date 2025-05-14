
from tqdm import tqdm

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_dataloader(src_dir, split, num_workers=8, batch_size=200, subset_size=None):
    dataset = datasets.ImageFolder(
        f"{src_dir}/{split}",
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]))
    if subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return loader


@torch.no_grad
def test(model, loader):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    total_correct = 0
    total_examples = 0
    for i, (images, target) in enumerate(tqdm(loader)):
        target = target.to(device=device, dtype=dtype)
        images = images.to(device=device, dtype=dtype)
        batch_size = images.shape[0]

        output = model(images)

        # measure accuracy
        pred = torch.argmax(output, axis=1)
        correct = (target == pred).sum()
        total_correct += int(correct)
        total_examples += int(batch_size)

    accuracy = 100 * (total_correct / total_examples)
    print(f"Accuracy: {accuracy:.2f}%, Total Correct: {total_correct}, Total Examples: {total_examples}")
    return accuracy, total_correct, total_examples
