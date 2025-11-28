import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

from default_config import ModelModes, ModelTypes, hific_args, directories
from src.model import Model
from src.helpers import utils
from src.compression.compression_utils import load_compressed_format

def generate_variations(args):
    # Ensure log directory exists
    log_dir = os.path.join(directories.experiments, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'generate_variations.log')

    # Setup logger and device
    logger = utils.logger_setup(logpath=log_path, filepath=os.path.abspath(__file__))
    device = utils.get_device()
    logger.info(f'Using device {device}')

    # Load the model
    if not args.model_path or not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

    _, model, _ = utils.load_model(
        args.model_path,
        logger=logger,
        device=device,
        model_mode=ModelModes.EVALUATION,
        strict=False, # Set to False if you are not loading all weights
        silent=True,
    )
    model.Hyperprior.hyperprior_entropy_model.build_tables()
    logger.info(f"Loaded model from {args.model_path}")

    model.eval() # Set model to evaluation mode

    # Read .hfc file
    compression_output = load_compressed_format(args.hfc_file)
    logger.info(f"Loaded .hfc file from {args.hfc_file}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate variations
    for i in range(args.num_variations):
        with torch.no_grad():
            reconstruction = model.decompress(compression_output, variation_strength=args.variation_strength)
        
        # Convert to PIL Image and save
        # Reconstruction is [0,1] range, (C,H,W) tensor
        reconstruction_img = transforms.ToPILImage()(reconstruction.cpu().squeeze(0))
        output_filename = f"variation_{os.path.basename(args.hfc_file).replace('.hfc', '')}_{i:03d}_strength_{args.variation_strength:.2f}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        reconstruction_img.save(output_path)
        logger.info(f"Saved variation {i+1}/{args.num_variations} to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image variations from an .hfc file.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--hfc_file", type=str, required=True,
                        help="Path to the input .hfc file.")
    parser.add_argument("--output_dir", type=str, default="variations_output",
                        help="Directory to save the generated variations.")
    parser.add_argument("--variation_strength", type=float, default=0.0,
                        help="Strength of the noise to add for variation. (e.g., 0.05, 0.1)")
    parser.add_argument("--num_variations", type=int, default=1,
                        help="Number of variations to generate.")
    
    args = parser.parse_args()
    generate_variations(args)
