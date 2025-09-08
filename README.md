# NVILA TinyOpenPSG Fine-tuning Repository

This repository contains code for fine-tuning NVILA (Neural Vision-Language Assistant) models on the OpenPSG dataset for visual relationship detection tasks. The project compares zero-shot performance against fine-tuned models and provides comprehensive evaluation tools.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a complete pipeline for:

1. **Data Preparation**: Converting OpenPSG dataset into fine-tuning format
2. **Model Fine-tuning**: Fine-tuning NVILA models using LoRA
3. **Inference**: Running inference on both pretrained and fine-tuned models
4. **Evaluation**: Comprehensive comparison between zero-shot and fine-tuned performance

The main goal is to evaluate how fine-tuning affects the model's ability to detect visual relationships between objects in images, particularly focusing on both closed-set and open-set scenarios.

## 📊 Dataset

**OpenPSG Dataset**: [Link to OpenPSG](https://github.com/Jingkang50/OpenPSG)

The OpenPSG dataset contains:
- Panoptic segmentation annotations
- Visual relationship triplets (subject, predicate, object)
- Support for both closed-set and open-set evaluation

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/transieu102/nvila_tinyopenpsg.git
cd nvila_tinyopenpsg
```

2. **Setup environment** (if needed):
```bash
bash env_setup.sh
```

## 📖 Usage

### 1. Data Preparation

Create fine-tuning data from the OpenPSG dataset:

```bash
python create_finetune_data.py
```

This script will:
- Load the OpenPSG dataset
- Create visualizations with colored object masks
- Generate both multi-choice and basic prompt formats
- Save processed data in `data/finetune/tiny_psg_data/`

**Key features**:
- Splits relations into base (80%) and novel (20%) sets
- Creates conversation-style training data
- Generates both multi-choice and free-form response formats

### 2. Model Fine-tuning

Run fine-tuning using the provided script:

```bash
bash /home/sieut/nvila_tinyopenpsg/scripts/sieut_openpgs/sft.sh
```

**Script parameters**:
- `RUN_NAME`: Experiment name (default: "NVILA-Lite-2B-finetune-tiny_openpsg_lora")
- `GLOBAL_TRAIN_BATCH_SIZE`: Global batch size (default: 2)
- `GRADIENT_ACCUMULATION_STEPS`: Gradient accumulation steps (default: 2)
- `MODELBASE`: Base model path (default: "Efficient-Large-Model/NVILA-Lite-2B")
- `DATA_MIXTURE`: Training data mixture (default: "TinyOpenPSGBasic+TinyOpenPSGMultipleChoice")

**Fine-tuning configuration**:
- Uses LoRA for efficient fine-tuning
- 1 training epoch
- Learning rate: 2e-5
- Vision tower: paligemma-siglip-so400m-patch14-448
- Only tunes the multimodal projector, not the vision tower or language model

### 3. Inference and Evaluation

Run inference on both pretrained and fine-tuned models:

```bash
python inference.py --config configs/eval_config.yaml
```

This will:
- Load both pretrained and fine-tuned models
- Run inference on validation data
- Save results to pickle file for further analysis

### 4. Experiment Analysis

#### Experiment 1: Classification Performance Analysis

```bash
python exp1.py --config configs/exp1.yaml
```

**What it does**:
- Compares zero-shot vs fine-tuned classification performance
- Generates confusion matrices
- Creates frequency vs F1 performance plots
- Supports both closed-set and open-set evaluation

**Outputs**:
- Classification reports (micro/macro precision, recall, F1)
- Confusion matrices (normalized)
- Class frequency vs F1 performance plots

#### Experiment 2: Embedding Similarity Analysis

```bash
python exp2.py --config configs/exp2.yaml
```

**What it does**:
- Analyzes embedding similarities between ground truth and predictions
- Creates threshold-recall curves
- Shows sample visualizations of best/worst cases
- Analyzes embedding diversity using PCA

**Outputs**:
- Threshold-recall curves
- Per-class recall analysis
- Sample visualizations
- PCA plots of embeddings
- Pairwise similarity distributions

## 📁 Project Structure

```
nvila_tinyopenpsg/
├── configs/                    # Configuration files
│   ├── eval_config.yaml       # Inference configuration
│   ├── exp1.yaml              # Experiment 1 configuration
│   └── exp2.yaml              # Experiment 2 configuration
├── scripts/                    # Training and utility scripts
│   └── sieut_openpgs/
│       └── sft.sh             # Fine-tuning script
├── llava/                      # LLaVA model implementation
├── data/                       # Data directories
│   ├── psg/                   # OpenPSG dataset
│   ├── coco/                  # COCO images
│   └── finetune/              # Processed fine-tuning data
├── results/                    # Experiment results
├── create_finetune_data.py    # Data preparation script
├── test_dataset.py            # Dataset utilities
├── inference.py               # Inference script
├── exp1.py                    # Classification analysis
├── exp2.py                    # Embedding analysis
└── README.md                  # This file
```

## ⚙️ Configuration

### Inference Configuration (`configs/eval_config.yaml`)

```yaml
# Model paths
pretrained_model: "path/to/pretrained/model"
finetuned_model: "path/to/finetuned/model"
finetuned_base: "path/to/base/model"

# Data paths
val_json: "data/psg/tiny_psg.json"
coco_dir: "data/coco"
train_json: "data/finetune/tiny_psg_data/tiny_psg_basic_data.json"

# Evaluation settings
max_samples: 1000  # Limit number of samples for testing
save_path: "results/inference_results.pkl"
```

### Experiment 1 Configuration (`configs/exp1.yaml`)

```yaml
clip_model: "ViT-B/32"
input_pickle: "results/inference_results.pkl"
is_open_set: true  # or false for closed-set evaluation
output:
  save_dir: "results/exp1_open"
  freq_vs_f1: "results/exp1_open/class_frequency_vs_f1_performance.png"
  confusion_zero: "results/exp1_open/zero_shot_confusion_matrix.png"
  confusion_finetune: "results/exp1_open/finetuned_confusion_matrix.png"
  metrics_report: "results/exp1_open/classification_report.txt"
```

### Experiment 2 Configuration (`configs/exp2.yaml`)

```yaml
clip_model: "ViT-B/32"
input_pickle: "results/inference_results.pkl"
subset: "open"  # or "closed"
output_dir: "exp2_outputs"
```

## 📈 Results

The experiments generate several types of results:

### Classification Metrics
- **Micro/Macro Precision, Recall, F1**: Overall performance metrics
- **Per-class F1**: Performance breakdown by relationship type
- **Confusion Matrices**: Detailed error analysis

### Visual Analysis
- **Frequency vs F1 Plots**: How class frequency affects performance
- **Sample Visualizations**: Best and worst case examples
- **PCA Embeddings**: Dimensionality reduction of model embeddings

### Similarity Analysis
- **Threshold-Recall Curves**: Performance at different similarity thresholds
- **Per-class Recall**: Recall analysis by relationship type
- **Embedding Diversity**: Analysis of embedding space characteristics

## 🔧 Key Features

### Data Processing
- **Automatic mask visualization**: Objects are highlighted with different colors
- **Multiple prompt formats**: Both multi-choice and free-form prompts
- **Base/Novel split**: 80/20 split for few-shot learning evaluation

### Model Training
- **LoRA fine-tuning**: Efficient parameter updates
- **Mixed data formats**: Combines basic and multi-choice training data
- **Configurable training**: Easy to modify hyperparameters

### Evaluation
- **Comprehensive metrics**: Multiple evaluation perspectives
- **Visual analysis**: Rich visualizations for understanding model behavior
- **Open/Closed set**: Separate evaluation for different scenarios

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 Notes

- The project uses NVILA-Lite-2B as the base model
- Fine-tuning is done using LoRA for efficiency
- Results are saved in pickle format for easy loading and analysis
- All visualizations are automatically generated and saved

## 🔗 Related Work

- [NVILA](https://github.com/Efficient-Large-Model/NVILA): Neural Vision-Language Assistant
- [OpenPSG](https://github.com/Jingkang50/OpenPSG): Open Panoptic Scene Graph Generation
- [LLaVA](https://github.com/haotian-liu/LLaVA): Large Language and Vision Assistant

---

For questions or issues, please open an issue in the repository.