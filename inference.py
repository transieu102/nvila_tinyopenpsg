"""
Predicate Classification Module for Visual Relationship Detection

This module provides tools for classifying relationships between objects in images
using LLaVA models with custom prompt templates.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import torch
from PIL import Image
from tqdm import tqdm
from test_dataset import TripletDataset
from llava import conversation as conversation_lib
import llava
from itertools import islice
import argparse
import yaml


@dataclass
class PredicateClassificationSample:
    """Data structure for predicate classification sample."""
    image_id: str
    triplet_id: str
    image: Image.Image
    subject_class: str
    object_class: str
    gt_predicate: str
    subject_mask_color: str
    object_mask_color: str
    gt_triplet: str
    is_open_set: bool


@dataclass
class PredicateClassificationResult(PredicateClassificationSample):
    """Result of predicate classification."""
    prompt_basic: str
    prompt_multi_choice: str
    pretrained_basic: str
    pretrained_multi_choice: str
    finetuned_basic: str
    finetuned_multi_choice: str
    



class PromptTemplates:
    """Template generator for predicate classification prompts."""
    
    MULTI_CHOICE_TEMPLATE = (
        "In this image, objects are marked with colors. "
        "Subject: {subject_class} [{subject_color}], Object: {object_class} [{object_color}]. "
        "Task: Output their relationship choose from: {predicate_classes}"
    )
    
    BASIC_TEMPLATE = (
        "In this image, objects are marked with colors. "
        "Subject: {subject_class} [{subject_color}], Object: {object_class} [{object_color}]. "
        "Task: Output their relationship as a short phrase (e.g., 'person holding apple')."
    )
    
    def __init__(self, predicate_classes: List[str]):
        """
        Initialize prompt templates.
        
        Args:
            predicate_classes: List of available predicate classes for multi-choice prompts
        """
        self.predicate_classes = predicate_classes
    
    def create_multi_choice_prompt(self, sample: PredicateClassificationSample) -> str:
        """Create a multi-choice prompt for the given sample."""
        return self.MULTI_CHOICE_TEMPLATE.format(
            subject_class=sample.subject_class,
            subject_color=sample.subject_mask_color,
            object_class=sample.object_class,
            object_color=sample.object_mask_color,
            predicate_classes=', '.join(self.predicate_classes)
        )
    
    def create_basic_prompt(self, sample: PredicateClassificationSample) -> str:
        """Create a basic prompt for the given sample."""
        return self.BASIC_TEMPLATE.format(
            subject_class=sample.subject_class,
            subject_color=sample.subject_mask_color,
            object_class=sample.object_class,
            object_color=sample.object_mask_color
        )


class ModelLoader:
    """Utility class for loading LLaVA models."""
    
    @staticmethod
    def load_model(
        model_path: str, 
        model_base: Optional[str] = None, 
        conv_mode: str = 'auto'
    ) -> Tuple[Any, Any]:
        """
        Load LLaVA model with specified configuration.
        
        Args:
            model_path: Path to the model
            model_base: Base model path (for fine-tuned models)
            conv_mode: Conversation mode
            
        Returns:
            Tuple of (model, generation_config)
        """
        # Setup conversation template
        conversation_lib.default_conversation = conversation_lib.conv_templates[conv_mode].copy()
        
        # Determine devices
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else ["cpu"]
        
        # Load model
        model = llava.load(model_path, model_base, devices=devices)
        
        # Get generation config
        generation_config = model.default_generation_config
        
        return model, generation_config


class PredicateClassifier:
    """Wrapper class for predicate classification using LLaVA models."""
    
    def __init__(
        self, 
        model_path: str, 
        model_base: Optional[str] = None, 
        conv_mode: str = 'auto'
    ):
        """
        Initialize the predicate classifier.
        
        Args:
            model_path: Path to the model
            model_base: Base model path (for fine-tuned models)
            conv_mode: Conversation mode
        """
        self.model_path = model_path
        self.model_base = model_base
        self.conv_mode = conv_mode
        
        # Load model
        self.model, self.generation_config = ModelLoader.load_model(
            model_path, model_base, conv_mode
        )
    
    def generate(self, image: Image.Image, prompt: str) -> str:
        """
        Generate response for given image and prompt.
        
        Args:
            image: Input image
            prompt: Text prompt
            
        Returns:
            Generated response string
        """
        response = self.model.generate_content(
            [image, prompt], 
            generation_config=self.generation_config
        )
        return response


class PredicateClassificationEvaluator:
    """Evaluator for comparing pretrained and fine-tuned models."""
    
    def __init__(
        self,
        pretrained_model_path: str,
        finetuned_model_path: str,
        finetuned_model_base: str,
        predicate_classes: List[str]
    ):
        """
        Initialize the evaluator.
        
        Args:
            pretrained_model_path: Path to pretrained model
            finetuned_model_path: Path to fine-tuned model
            finetuned_model_base: Base model for fine-tuned model
            predicate_classes: List of predicate classes
        """
        self.pretrained_classifier = PredicateClassifier(pretrained_model_path)
        self.finetuned_classifier = PredicateClassifier(
            finetuned_model_path, 
            finetuned_model_base
        )
        self.prompt_templates = PromptTemplates(predicate_classes)
    
    def evaluate_sample(self, sample: PredicateClassificationSample) -> PredicateClassificationResult:
        """
        Evaluate a single sample with both models and both prompt types.
        
        Args:
            sample: Sample to evaluate
            
        Returns:
            Dictionary containing all responses and ground truth
        """
        # Create prompts
        basic_prompt = self.prompt_templates.create_basic_prompt(sample)
        multi_choice_prompt = self.prompt_templates.create_multi_choice_prompt(sample)
        
        # Generate responses
        results = {
            'prompt_basic': basic_prompt,
            'prompt_multi_choice': multi_choice_prompt,
            'pretrained_basic': self.pretrained_classifier.generate(sample.image, basic_prompt),
            'pretrained_multi_choice': self.pretrained_classifier.generate(sample.image, multi_choice_prompt),
            'finetuned_basic': self.finetuned_classifier.generate(sample.image, basic_prompt),
            'finetuned_multi_choice': self.finetuned_classifier.generate(sample.image, multi_choice_prompt)
        }
        results = PredicateClassificationResult(**asdict(sample), **results)
        
        return results
    
    def print_results(self, results: PredicateClassificationResult) -> None:
        """Print evaluation results in a formatted way."""
        print("=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Ground Truth Predicate: {results.gt_predicate}")
        print(f"Ground Truth Triplet: {results.gt_triplet}")
        print("-" * 30)
        print("PRETRAINED MODEL:")
        print(f"  Basic prompt response: {results.pretrained_basic}")
        print(f"  Multi-choice response: {results.pretrained_multi_choice}")
        print("-" * 30) 
        print("FINE-TUNED MODEL:")
        print(f"  Basic prompt response: {results.finetuned_basic}")
        print(f"  Multi-choice response: {results.finetuned_multi_choice}")
        print("=" * 60)


def load_training_relations(train_json_path: str) -> List[str]:
    """
    Load relation classes from training data.
    
    Args:
        train_json_path: Path to training JSON file
        
    Returns:
        List of unique relation classes from training data
    """
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    return list(set([
        sample['conversations'][1]['value'] 
        for sample in train_data
    ]))


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_config(config: dict, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

def main(config_path: str):
    """Main evaluation function."""
    # Load configuration
    config = load_config(config_path)

    save_dir = Path(config['save_path']).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training relations
    closed_relations = load_training_relations(config['train_json'])
    
    # Create dataset
    dataset = TripletDataset(
        config['val_json'], 
        config['coco_dir'], 
        split='val', 
        # unsupport_rels=closed_relations
    )
    if 'max_samples' in config:
        max_samples = min(config['max_samples'], len(dataset))
    else:
        max_samples = len(dataset)
        config['max_samples'] = max_samples
    
    # Create evaluator
    evaluator = PredicateClassificationEvaluator(
        pretrained_model_path=config['pretrained_model'],
        finetuned_model_path=config['finetuned_model'],
        finetuned_model_base=config['finetuned_base'],
        predicate_classes=dataset.rel_cats
    )
    all_results = []
    # Evaluate samples
    for sample_data in tqdm(islice(dataset, max_samples), total=max_samples):
        sample = PredicateClassificationSample(**sample_data)
        sample.is_open_set = sample.gt_predicate not in closed_relations
        
        results = evaluator.evaluate_sample(sample)
        all_results.append(results)
    
    # Save results
    with open(config['save_path'], 'wb') as f:
        pickle.dump(all_results, f)
    
    # Save (possibly updated) config back to yaml
    save_config(config, save_dir / "config.yaml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predicate classification evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    main(args.config)