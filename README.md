

# Visual Wings for Textual Prompts: Infusing Textual Prompts with Visual Features for Open-set Recognition

Code for our paper: Visual Wings for Textual Prompts: Infusing Textual Prompts with Visual Features for Open-set Recognition.

![image](https://github.com/FayeXXX/INFUSE/blob/main/framework.png)

## Data preparation

We follow the data setting as in [osr_closed_set_all_you_need](https://github.com/sgvaze/osr_closed_set_all_you_need) .

Download datasets below:    



[CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html),

[TinyImageNet](https://github.com/rmccorm4/Tiny-Imagenet-200)

For TinyImageNet, you also need to run `create_val_img_folder` in `data/tinyimagenet.py` to create
a directory with the test data.

[LSUN](https://github.com/facebookresearch/odin)


## Installation

* Environments

```
# Create a conda environment
conda create -y -n infuse python=3.8

# Activate the environment
conda activate infuse

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

* Vision-Language models downloading

  "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
  "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"

* configuration

  Set paths to datasets and pre-trained models (for fine-grained experiments) in ```config.py```

### Training

```
bash run_infuse.sh
```

**hyper-parameters:**

| **Dataset**       | **Image Size** | **Learning Rate** | **Batch Size** | Epoch |
|---------------|------------|---------------|------------|------------|
| CIFAR-10      | 32      | 0.003       | 100      | 60    |
| CIFAR + N     | 32      | 0.003        | 100      | 60    |
| TinyImageNet  | 64         | 0.003        | 64      | 60    |

## Acknowledgements

Our code is based on [CoOp](https://github.com/KaiyangZhou/CoOp) ,[osr_closed_set_all_you_need](https://github.com/sgvaze/osr_closed_set_all_you_need) , and [cat](https://github.com/linhezheng19/CAT) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.