"""
Image Data Preprocessing Module
Handles loading, filtering, and preprocessing of satellite imagery data for fire risk classification.
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class FireRiskDataPreprocessor:
    """Handles preprocessing of FireRisk dataset"""
    
    def __init__(self, dataset_name="blanchon/FireRisk", target_samples_per_class=1300):
        self.dataset_name = dataset_name
        self.target_samples_per_class = target_samples_per_class
        
        # Class mapping to consolidate similar classes
        self.class_mapping = {
            0: 0,  # High stays 0
            1: 1,  # Low stays 1
            2: 2,  # Moderate stays 2
            3: 3,  # Non-burnable stays 3
            4: 0,  # Very-high maps to High (0)
            5: 1,  # Very-low maps to Low (1)
            # Class 6 (Water) will be filtered out
        }
        
        self.class_names = ["High", "Low", "Moderate", "Non-burnable"]
        
    def load_dataset(self):
        """Load the FireRisk dataset from Hugging Face"""
        logger.info(f"Loading dataset: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name)
        return dataset
    
    def print_class_distribution(self, dataset_split, split_name):
        """Print class distribution for a dataset split"""
        labels = [example['label'] for example in dataset_split]
        counts = pd.Series(labels).value_counts().sort_index()
        
        logger.info(f"\n{split_name} set distribution:")
        for idx, count in counts.items():
            class_name = self.class_names[idx] if idx < len(self.class_names) else f"Class {idx}"
            logger.info(f"  {class_name} (Class {idx}): {count} images")
        
        return counts
    
    def filter_and_map_classes(self, dataset_split):
        """Filter out water class (6) and map remaining classes"""
        # Filter out class 6 (Water) and apply mapping
        filtered_examples = []
        
        for example in dataset_split:
            if example['label'] != 6:  # Remove water class
                # Apply class mapping
                example['label'] = self.class_mapping[example['label']]
                filtered_examples.append(example)
        
        logger.info(f"Filtered out water class. Remaining samples: {len(filtered_examples)}")
        return filtered_examples
    
    def balance_classes(self, dataset_split):
        """Balance classes by sampling equal numbers from each class"""
        # Group examples by class
        class_examples = defaultdict(list)
        for i, example in enumerate(dataset_split):
            class_examples[example['label']].append(i)
        
        # Sample equal amounts from each class
        balanced_indices = []
        for label, indices in class_examples.items():
            if len(indices) > self.target_samples_per_class:
                # Random sampling
                sampled_indices = random.sample(indices, self.target_samples_per_class)
                balanced_indices.extend(sampled_indices)
            else:
                # Use all available samples
                balanced_indices.extend(indices)
        
        # Shuffle indices
        random.shuffle(balanced_indices)
        
        # Create balanced dataset
        balanced_data = [dataset_split[i] for i in balanced_indices]
        
        logger.info(f"Balanced dataset size: {len(balanced_data)}")
        return balanced_data
    
    def preprocess_splits(self, dataset):
        """Preprocess all splits of the dataset"""
        processed_dataset = {}
        
        for split_name in dataset.keys():
            logger.info(f"\nProcessing {split_name} split...")
            
            # Convert to list for easier manipulation
            split_data = list(dataset[split_name])
            
            # Print original distribution
            self.print_class_distribution(split_data, f"Original {split_name}")
            
            # Filter and map classes
            filtered_data = self.filter_and_map_classes(split_data)
            
            # Balance classes
            target_samples = self.target_samples_per_class if split_name == 'train' else 300
            self.target_samples_per_class = target_samples
            balanced_data = self.balance_classes(filtered_data)
            
            # Print final distribution
            self.print_class_distribution(balanced_data, f"Processed {split_name}")
            
            processed_dataset[split_name] = balanced_data
            
        return processed_dataset
    
    def save_processed_data(self, processed_dataset, output_dir):
        """Save processed data to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, data in processed_dataset.items():
            # Save as pickle for Python objects
            import pickle
            file_path = output_path / f"{split_name}_processed.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved {split_name} split to {file_path}")
            
            # Also save metadata
            metadata = {
                'split_name': split_name,
                'num_samples': len(data),
                'class_distribution': pd.Series([item['label'] for item in data]).value_counts().to_dict(),
                'class_names': self.class_names
            }
            
            metadata_path = output_path / f"{split_name}_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Preprocess FireRisk dataset')
    parser.add_argument('--dataset', default='blanchon/FireRisk', help='Dataset name')
    parser.add_argument('--output', default='data/processed/', help='Output directory')
    parser.add_argument('--samples-per-class', type=int, default=1300, help='Target samples per class for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    set_seeds(args.seed)
    
    # Initialize preprocessor
    preprocessor = FireRiskDataPreprocessor(
        dataset_name=args.dataset,
        target_samples_per_class=args.samples_per_class
    )
    
    # Load and preprocess dataset
    dataset = preprocessor.load_dataset()
    processed_dataset = preprocessor.preprocess_splits(dataset)
    
    # Save processed data
    preprocessor.save_processed_data(processed_dataset, args.output)
    
    logger.info("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()
