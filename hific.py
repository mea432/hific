import os
import glob
import time
import argparse
import torch
import torchvision
from PIL import Image

from src.helpers import utils
from src.compression import compression_utils
from default_config import ModelModes

class DummyLogger:
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass

def load_model(ckpt_path):
    """Loads the model from a checkpoint."""
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
    model.Hyperprior.hyperprior_entropy_model.build_tables()
    return model

def compress(input_path, output_path, ckpt_path):
    """Compresses an image or a directory of images."""
    model = load_model(ckpt_path)
    device = next(model.parameters()).device

    if os.path.isdir(input_path):
        input_files = glob.glob(os.path.join(input_path, '*.png'))
        input_files += glob.glob(os.path.join(input_path, '*.jpg'))
    else:
        input_files = [input_path]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_path in input_files:
        img = Image.open(file_path).convert('RGB')
        tensor = torchvision.transforms.ToTensor()(img).unsqueeze(0).to(device)

        with torch.no_grad():
            compressed_output = model.compress(tensor)

        base_name = os.path.basename(file_path)
        file_name, _ = os.path.splitext(base_name)
        output_file = os.path.join(output_path, f'{file_name}.hfc')

        actual_bpp, theoretical_bpp = compression_utils.save_compressed_format(
            compressed_output, output_file
        )
        print(f'Compressed {file_path} to {output_file}')
        print(f'  - Actual BPP: {actual_bpp:.4f}')
        print(f'  - Theoretical BPP: {theoretical_bpp:.4f}')


def decompress(input_path, output_path, ckpt_path):
    """Decompresses a .hfc file or a directory of .hfc files."""
    model = load_model(ckpt_path)

    if os.path.isdir(input_path):
        input_files = glob.glob(os.path.join(input_path, '*.hfc'))
    else:
        input_files = [input_path]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_path in input_files:
        compressed_output = compression_utils.load_compressed_format(file_path)

        with torch.no_grad():
            reconstruction = model.decompress(compressed_output)

        base_name = os.path.basename(file_path)
        file_name, _ = os.path.splitext(base_name)
        output_file = os.path.join(output_path, f'{file_name}_reconstructed.png')
        
        torchvision.utils.save_image(reconstruction, output_file, normalize=True)
        print(f'Decompressed {file_path} to {output_file}')

def main():
    parser = argparse.ArgumentParser(description='High-Fidelity Generative Compression')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Compress command
    compress_parser = subparsers.add_parser('compress', help='Compress an image or a directory of images')
    compress_parser.add_argument('-i', '--input_path', type=str, required=True, help='Path to an image or a directory of images')
    compress_parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to the output directory for compressed files')
    compress_parser.add_argument('--ckpt_path', type=str, default='hific_low.pt', help='Path to the model checkpoint')

    # Decompress command
    decompress_parser = subparsers.add_parser('decompress', help='Decompress a .hfc file or a directory of .hfc files')
    decompress_parser.add_argument('-i', '--input_path', type=str, required=True, help='Path to a .hfc file or a directory of .hfc files')
    decompress_parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to the output directory for reconstructed images')
    decompress_parser.add_argument('--ckpt_path', type=str, default='hific_low.pt', help='Path to the model checkpoint')

    args = parser.parse_args()

    if args.command == 'compress':
        compress(args.input_path, args.output_path, args.ckpt_path)
    elif args.command == 'decompress':
        decompress(args.input_path, args.output_path, args.ckpt_path)

if __name__ == '__main__':
    main()
