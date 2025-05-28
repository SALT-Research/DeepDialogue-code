# DeepDialogue

This repository hosts the code and data for the paper:

> **DeepDialogue: A Multi-Turn Emotionally-Rich Spoken Dialogue Dataset**  
> [Project Website](https://salt-research.github.io/DeepDialogue)

[![paper](https://img.shields.io/badge/Paper-arXiv-green)](https://arxiv.org/abs/2505.19978)

## 📁 Code and Data

### 🔧 Code
The code to replicate our experiments on the speech emotion recognition (SER) task can be found in the `ser/` directory.
You can run the code using the following command:

```bash
bash train.sh
```
This will execute the training script with the default parameters. You can modify the `train.sh` file to change the parameters as needed.

To evaluate a pre-trained model on the RAVDESS dataset, use the evaluation script:

```bash
python evaluate_ravdess.py \
    --model_path /path/to/deepdialogue-trained/model \
    --ssl_model_name facebook/hubert-base-ls960 \
    --ravdess_root /path/to/ravdess \
    --hidden_size 128 \
    --feature_dim 768 \
    --batch_size 32 \
    --seed 42
```

This script performs 5-fold cross-validation on the RAVDESS dataset (with no specific training on RAVDESS dataset) and provides detailed accuracy and F1-score metrics. `/path/to/deepdialogue-trained/model` should be replaced with the path to your pre-trained model, and `/path/to/ravdess` should be replaced with the folder containing RAVDESS data in the standard format (e.g., `Actor_01/03-01-01-01-01-01-01.wav`).

### 📚 Data
The dataset is available in two versions, both on 🤗 HuggingFace:
  - [🤗 XTTS version](https://huggingface.co/datasets/SALT-Research/DeepDialogue-xtts)
  - [🤗 Orpheus version](https://huggingface.co/datasets/SALT-Research/DeepDialogue-orpheus)


## 📣 Citation

If you use this dataset in your research, please cite our [paper](https://arxiv.org/abs/2505.19978):

```
@misc{koudounas2025deepdialoguemultiturnemotionallyrichspoken,
      title={DeepDialogue: A Multi-Turn Emotionally-Rich Spoken Dialogue Dataset}, 
      author={Alkis Koudounas and Moreno La Quatra and Elena Baralis},
      year={2025},
      eprint={2505.19978},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.19978}, 
}
```
