# HiFiC: High-Fidelity Generative Image Compression (PyTorch)

This repository contains a PyTorch implementation of the paper ["High-Fidelity Generative Image Compression" by Mentzer et. al.](https://hific.github.io/). It provides a command-line tool to compress and decompress images using pre-trained models, as well as the ability to train new models.

The model is capable of compressing images to a fraction of their original size while maintaining high perceptual quality. The outputs are often more visually pleasing than standard codecs like JPEG at similar or even lower bitrates.

For the official TensorFlow implementation, see the [TensorFlow compression repo](https://github.com/tensorflow/compression/tree/master/models/hific).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Justin-Tan/high-fidelity-generative-compression/blob/master/assets/HiFIC_torch_colab_demo.ipynb)

## Important Note on Usage

The generator is trained for realism, not for perfect reconstruction. It may synthesize textures and details to remove compression artifacts. This means that **reconstructed images are not guaranteed to be identical to the input**.

## Prerequisites

Before you begin, ensure you have the following software installed:

* [Git](https://git-scm.com/downloads)
* [Python](https://www.python.org/downloads/) (3.8+ recommended)
* [CMake](https://cmake.org/download/)
* [PyTorch](https://pytorch.org/) (Follow instructions on their website for your specific system)

## 1. Setup

First, clone the repository and navigate into the project directory:

```bash
git clone https://github.com/mea432/hific.git
cd hific
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

Pretrained model weights using the OpenImages dataset can be found below (~2 GB). The examples at the end of this readme were produced using the HIFIC-med model. The same models are also hosted in the following Zenodo repository: <https://zenodo.org/record/4026003>.

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

* You can also provide a directory for the `-i` argument to compress all PNG/JPG images inside it.
* Use different models (`hific_low.pt`, `hific_med.pt`, `hific_high.pt`) to target different bitrates.

### Decompressing an Image

To decompress a `.hfc` file, use the `decompress` command. This will reconstruct the image as a `.png` file.

```bash
python hific.py decompress \
  -i path/to/your/compressed.hfc \
  -o path/to/reconstruction/directory \
  --ckpt_path pretrained_models/hific_med.pt
```

* The same model that was used for compression must be used for decompression.
* You can also provide a directory for the `-i` argument to decompress all `.hfc` files inside it.

## Example

Here is an example of an original image and its reconstruction using HiFiC.

Original | HiFIC
:-------------------------:|:-------------------------:
![guess](assets/originals/CLIC2020_5.png) | ![guess](assets/hific/CLIC2020_5_RECON_0.160bpp.png)

```
Original: (6.01 bpp - 2100 kB) | HiFIC: (0.160 bpp - 56 kB). Ratio: 37.5.
```

More examples can be found in [assets/EXAMPLES.md](assets/EXAMPLES.md).

## Acknowledgements

* The compression routines under `src/compression/` are derived from the [Tensorflow Compression library](https://github.com/tensorflow/compression).
* The vectorized rANS implementation is based on [Craystack](https://github.com/j-towns/craystack).
* The perceptual loss code (`src/loss/perceptual_similarity/`) is based on the [Perceptual Similarity repository](https://github.com/richzhang/PerceptualSimilarity).

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

