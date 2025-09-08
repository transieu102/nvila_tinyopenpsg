import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from test_dataset import TripletDataset
import random
class PSGDataProcessor:
    """Class to process PSG dataset and generate fine-tuning data."""
    
    def __init__(self, json_file: str, coco_dir: str, save_dir: str = 'data/finetune/tiny_psg_data', split: str = 'train', support_rels: list = None):
        """
        Initialize the PSG data processor.
        
        Args:
            json_file: Path to PSG JSON file
            coco_dir: Path to COCO dataset directory
            save_dir: Directory to save processed data
            split: Split to process
            support_rels: List of support relations
        """
        self.json_file = json_file
        self.coco_dir = coco_dir
        self.save_dir = Path(save_dir)
        self.image_dir = self.save_dir / 'images'
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.support_rels = support_rels
        # Initialize dataset
        self.dataset = TripletDataset(json_file, coco_dir, split=split, support_rels=support_rels)
        self.predicate_classes = self.dataset.rel_cats
        
        # Initialize data containers
        self.multi_choice_data = []
        self.basic_data = []
    
    @property
    def multi_choice_prompt_template(self) -> str:
        """Template for multi-choice prompts."""
        return (
            "In this image, objects are marked with colors. "
            "Subject: {subject_class} [{subject_color}], Object: {object_class} [{object_color}]. "
            "Task: Output their relationship choose from: {predicate_classes}"
        )
    
    @property
    def basic_prompt_template(self) -> str:
        """Template for basic prompts."""
        return (
            "In this image, objects are marked with colors. "
            "Subject: {subject_class} [{subject_color}], Object: {object_class} [{object_color}]. "
            "Task: Output their relationship as a short phrase (e.g., 'person holding apple')."
        )
    
    def _create_multi_choice_prompt(self, sample: Dict[str, Any]) -> str:
        """Create multi-choice prompt for a sample."""
        return self.multi_choice_prompt_template.format(
            subject_class=sample['subject_class'],
            subject_color=sample.get('subject_mask_color', ''),
            object_class=sample['object_class'],
            object_color=sample.get('object_mask_color', ''),
            predicate_classes=self.predicate_classes
        )
    
    def _create_basic_prompt(self, sample: Dict[str, Any]) -> str:
        """Create basic prompt for a sample."""
        return self.basic_prompt_template.format(
            subject_class=sample['subject_class'],
            subject_color=sample.get('subject_mask_color', ''),
            object_class=sample['object_class'],
            object_color=sample.get('object_mask_color', '')
        )
    
    def _save_image(self, sample: Dict[str, Any]) -> str:
        """Save image and return relative path."""
        triplet_id = sample['triplet_id']
        image_filename = f"{triplet_id}.jpg"
        image_path = self.image_dir / image_filename
        #check if path exists
        if image_path.exists():
            return f"images/{image_filename}"
        sample['image'].save(str(image_path))
        return f"images/{image_filename}"
    
    def _create_conversation_entry(self, image_path: str, prompt: str, response: str) -> Dict[str, Any]:
        """Create a conversation entry for fine-tuning data."""
        return {
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": prompt
                },
                {
                    "from": "gpt",
                    "value": response
                }
            ]
        }
    
    def _process_sample(self, sample: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Process a single sample and return multi-choice and basic data entries."""
        # Save image
        image_path = self._save_image(sample)
        
        # Create prompts
        multi_choice_prompt = self._create_multi_choice_prompt(sample)
        basic_prompt = self._create_basic_prompt(sample)
        
        # Create responses
        multi_choice_response = sample['gt_predicate']
        basic_response = " ".join([
            sample['subject_class'],
            sample['gt_predicate'],
            sample['object_class']
        ])
        
        # Create conversation entries
        multi_choice_entry = self._create_conversation_entry(
            image_path, multi_choice_prompt, multi_choice_response
        )
        basic_entry = self._create_conversation_entry(
            image_path, basic_prompt, basic_response
        )
        
        return multi_choice_entry, basic_entry
    
    def process_dataset(self, limit: int = None) -> None:
        """
        Process the entire dataset.
        
        Args:
            limit: Optional limit on number of samples to process
        """
        for i, sample in enumerate(self.dataset):
            if limit and i >= limit:
                break
                
            print(f"Processing sample {i + 1}: {sample.get('triplet_id', 'unknown')}")
            
            multi_choice_entry, basic_entry = self._process_sample(sample)
            
            self.multi_choice_data.append(multi_choice_entry)
            self.basic_data.append(basic_entry)
    
    def save_data(self, 
                  multi_choice_filename: str = 'tiny_psg_multi_choice_data.json',
                  basic_filename: str = 'tiny_psg_basic_data.json') -> None:
        """
        Save processed data to JSON files.
        
        Args:
            multi_choice_filename: Filename for multi-choice data
            basic_filename: Filename for basic data
        """
        # Save multi-choice data
        multi_choice_path = self.save_dir / multi_choice_filename
        with open(multi_choice_path, 'w', encoding='utf-8') as f:
            json.dump(self.multi_choice_data, f, indent=2, ensure_ascii=False)
        
        # Save basic data
        basic_path = self.save_dir  / basic_filename
        with open(basic_path, 'w', encoding='utf-8') as f:
            json.dump(self.basic_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.multi_choice_data)} multi-choice samples to {multi_choice_path}")
        print(f"Saved {len(self.basic_data)} basic samples to {basic_path}")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            'total_samples': len(self.multi_choice_data),
            'predicate_classes': len(self.support_rels),
            'images_saved': len(list(self.image_dir.glob('*.jpg')))
        }


def main():
    """Main function to run the PSG data processing."""
    # Configuration
    random.seed(42)
    json_file = 'data/psg/tiny_psg.json'
    coco_dir = 'data/coco'
    save_dir = 'data/finetune/tiny_psg_data'
    with open(json_file, "r") as f:
        data = json.load(f)
        all_rel_cats = data["predicate_classes"]
    novel_ratio = 0.2
    n_novel = int(len(all_rel_cats) * novel_ratio)
    novel_rels = set(random.sample(all_rel_cats, n_novel))
    base_rels = set(all_rel_cats) - novel_rels 
    
    print(f"Base rels: {base_rels}")
    print(f"Novel rels: {novel_rels}")
    # for subset in ['train', 'val_basic', 'val_novel']:
        # Initialize processor
    processor = PSGDataProcessor(json_file, coco_dir, save_dir, split='train', support_rels=list(base_rels))
    # Process dataset (limit to 1 sample for testing, remove limit for full processing)
    # processor.process_dataset(limit=1)
    processor.process_dataset(limit=None)
    # Save data
    processor.save_data(
        multi_choice_filename=f'tiny_psg_multi_choice_data.json',
        basic_filename=f'tiny_psg_basic_data.json'
    )
    
    # Print statistics
    stats = processor.get_statistics()
    print("\nProcessing Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()