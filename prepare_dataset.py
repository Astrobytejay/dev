from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor
import os
import torch

# Load your dataset
dataset = load_dataset("bitmind/celeb-a-hq___0-to-4999___FLUX.1-dev")

# Define image transformation pipeline
transform = Compose([
    Resize((512, 512)),  # Resizing images to 512x512
    ToTensor(),          # Converting images to tensors
])

# Apply transformation to your dataset and save it
def save_transformed_dataset(dataset, split):
    os.makedirs(f'/workspace/transformed_dataset/{split}', exist_ok=True)
    for i, item in enumerate(dataset):
        image = transform(item['image'])
        label = item['label']
        torch.save((image, label), f'/workspace/transformed_dataset/{split}/{i}.pt')

save_transformed_dataset(dataset['train'], 'train')
