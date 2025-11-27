
import os
import glob
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image

from src.helpers import utils
from src.compression import compression_utils
from default_config import ModelModes

class DummyLogger:
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass

def load_hific_model(ckpt_path):
    """Loads the HiFiC model from a checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = DummyLogger()
    _, model, _ = utils.load_model(
        ckpt_path,
        logger=logger,
        device=device,
        model_mode=ModelModes.EVALUATION,
        strict=False,
        silent=True,
    )
    if model is not None:
        model.Hyperprior.hyperprior_entropy_model.build_tables()
    return model

class HFCClassificationDataset(Dataset):
    """
    A PyTorch Dataset for loading and decompressing images from .hfc format.
    """

    def __init__(self, root_dir, ckpt_path, transform=None):
        self.root_dir = root_dir
        self.hific_model = load_hific_model(ckpt_path)
        self.transform = transform
        self.label_map = {}
        self.class_names = []

        if self.hific_model is None:
            raise RuntimeError("Could not load HiFiC model. Please check the checkpoint path.")

        # Find all .hfc files
        all_hfc_files = glob.glob(os.path.join(root_dir, '*.hfc'))
        
        # Populate class_names and label_map from filenames
        current_label_id = 0
        self.hfc_files = [] # Only include files for which a label can be extracted
        for f in all_hfc_files:
            base_name_without_ext = os.path.splitext(os.path.basename(f))[0]
            
            # Extract class name from filename
            parts = base_name_without_ext.split('_')
            if len(parts) > 2: # Ensure there are enough parts to extract the class
                class_name = parts[2] # e.g., 'S-aureus'
                if class_name not in self.class_names:
                    self.class_names.append(class_name)
                    self.label_map[class_name] = current_label_id
                    current_label_id += 1
                
                # Assign the numeric label to the file
                self.hfc_files.append(f)

        # Create numerical labels list for all files
        self.labels = []
        for f in self.hfc_files:
            base_name_without_ext = os.path.splitext(os.path.basename(f))[0]
            parts = base_name_without_ext.split('_')
            class_name = parts[2]
            self.labels.append(self.label_map[class_name])
            
        if not self.hfc_files:
            raise RuntimeError("No valid .hfc files found in the directory or no class names could be extracted.")

    def __len__(self):
        return len(self.hfc_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        hfc_path = self.hfc_files[idx]
        
        try:
            # Decompress the image
            compressed_output = compression_utils.load_compressed_format(hfc_path)
            with torch.no_grad():
                reconstruction = self.hific_model.decompress(compressed_output)
            
            # The output is a tensor, but transforms often expect a PIL image
            # Convert tensor to PIL Image
            image = torchvision.transforms.ToPILImage()(reconstruction.squeeze(0))

            if self.transform:
                image = self.transform(image)

            label = self.labels[idx]
            return image, label

        except Exception as e:
            print(f"Error loading or decompressing {hfc_path}: {e}")
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_hfc_dataloaders(root_dir, ckpt_path, batch_size=32, shuffle=True, transform=None, num_workers=4, pin_memory=True):
    """
    Creates a DataLoader for the HFCClassificationDataset.
    """
    if transform is None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = HFCClassificationDataset(root_dir=root_dir, ckpt_path=ckpt_path, transform=transform)
    

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    return dataloader

