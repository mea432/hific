# HiFiC: High-Fidelity Generative Image Compression (PyTorch)

This repository contains a PyTorch implementation of the paper ["High-Fidelity Generative Image Compression" by Mentzer et. al.](https://hific.github.io/). It provides a command-line tool to compress and decompress images using pre-trained models, as well as the ability to train new models.

The model is capable of compressing images to a fraction of their original size while maintaining high perceptual quality. The outputs are often more visually pleasing than standard codecs like JPEG at similar or even lower bitrates.

For the official TensorFlow implementation, see the [TensorFlow compression repo](https://github.com/tensorflow/compression/tree/master/models/hific).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Justin-Tan/high-fidelity-generative-compression/blob/master/assets/HiFIC_torch_colab_demo.ipynb)

## Important Note on Usage
The generator is trained for realism, not for perfect reconstruction. It may synthesize textures and details to remove compression artifacts. This means that **reconstructed images are not guaranteed to be identical to the input**.

> "_Therefore, we emphasize that our method is not suitable for sensitive image contents, such as, e.g., storing medical images, or important documents._" — Original Authors

## Prerequisites
Before you begin, ensure you have the following software installed:
*   [Git](https://git-scm.com/downloads)
*   [Python](https://www.python.org/downloads/) (3.8+ recommended)
*   [CMake](https://cmake.org/download/)
*   [PyTorch](https://pytorch.org/) (Follow instructions on their website for your specific system)

## 1. Setup

First, clone the repository and navigate into the project directory:
```bash
git clone https://github.com/Justin-Tan/high-fidelity-generative-compression.git
cd high-fidelity-generative-compression
```

Next, install the required Python packages:
```bash
pip install -r requirements.txt
```

Finally, build the C++ extension required for entropy coding. This command compiles and installs the extension in your current environment.
```bash
pip install .
```
To verify your setup, you can run `python hific.py --help`.

## 2. Download Pre-trained Models

Pre-trained models are available for different bitrate targets. We recommend using a tool like `gdown` to download them from Google Drive.

First, install `gdown`:
```bash
pip install gdown
```

Then, download the models into the `pretrained_models` directory:
```bash
mkdir -p pretrained_models

# Low bitrate model (~0.14 bpp)
gdown --id 1hfFTkZbs_VOBmXQ-M4bYEPejrD76lAY9 -O pretrained_models/hific_low.pt

# Medium bitrate model (~0.30 bpp)
gdown --id 1QNoX0AGKTBkthMJGPfQI0dT0_tnysYUb -O pretrained_models/hific_med.pt

# High bitrate model (~0.45 bpp)
gdown --id 1BFYpvhVIA_Ek2QsHBbKnaBE8wn1GhFyA -O pretrained_models/hific_high.pt
```

## 3. Usage

This tool provides `compress` and `decompress` commands.

### Compressing an Image

To compress an image, use the `compress` command. The output will be a `.hfc` file.

```bash
python hific.py compress \
  -i path/to/your/image.png \
  -o path/to/output/directory \
  --ckpt_path pretrained_models/hific_med.pt
```
*   You can also provide a directory for the `-i` argument to compress all PNG/JPG images inside it.
*   Use different models (`hific_low.pt`, `hific_med.pt`, `hific_high.pt`) to target different bitrates.

### Decompressing an Image

To decompress a `.hfc` file, use the `decompress` command. This will reconstruct the image as a `.png` file.

```bash
python hific.py decompress \
  -i path/to/your/compressed.hfc \
  -o path/to/reconstruction/directory \
  --ckpt_path pretrained_models/hific_med.pt
```
*   The same model that was used for compression must be used for decompression.
*   You can also provide a directory for the `-i` argument to decompress all `.hfc` files inside it.

## Example
Here is an example of an original image and its reconstruction using HiFiC.

Original | HiFIC
:-------------------------:|:-------------------------:
![guess](assets/originals/CLIC2020_5.png) | ![guess](assets/hific/CLIC2020_5_RECON_0.160bpp.png)

```
Original: (6.01 bpp - 2100 kB) | HiFIC: (0.160 bpp - 56 kB). Ratio: 37.5.
```
More examples can be found in [assets/EXAMPLES.md](assets/EXAMPLES.md).

## Advanced Usage: Training

It is also possible to train your own models.

1.  **Dataset:** Download a large (>100,000) dataset of diverse color images (e.g., [OpenImages](https://storage.googleapis.com/openimages/web/index.html)). Add the dataset path in `default_config.py`.

2.  **Base Model Training:** First, train a base model with only the rate-distortion loss.
    ```bash
    # Train initial autoencoding model
    python3 train.py --model_type compression --regime low --n_steps 1e6
    ```

3.  **GAN Training:** Then, use the checkpoint of the trained base model to 'warmstart' the GAN training.
    ```bash
    # Train using full generator-discriminator loss
    python3 train.py --model_type compression_gan --regime low --n_steps 1e6 --warmstart --ckpt path/to/base/checkpoint
    ```

## Acknowledgements
*   The compression routines under `src/compression/` are derived from the [Tensorflow Compression library](https://github.com/tensorflow/compression).
*   The vectorized rANS implementation is based on [Craystack](https://github.com/j-towns/craystack).
*   The perceptual loss code (`src/loss/perceptual_similarity/`) is based on the [Perceptual Similarity repository](https://github.com/richzhang/PerceptualSimilarity).

## Citation
This is a PyTorch port of the original work. Please cite the original paper if you use this code.
```
@article{mentzer2020high,
  title={High-Fidelity Generative Image Compression},
  author={Mentzer, Fabian and Toderici, George and Tschannen, Michael and Agustsson, Eirikur},
  journal={arXiv preprint arXiv:2006.09965},
  year={2020}
}
```

## License
This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.