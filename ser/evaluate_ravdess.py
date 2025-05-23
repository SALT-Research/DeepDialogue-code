import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import glob
import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoFeatureExtractor, AutoModel
import torch.nn as nn
import argparse
from tqdm import tqdm
import json
from datetime import datetime


class SSLMLPModel(nn.Module):
    """
    SSL (Self-Supervised Learning) + MLP model for emotion classification.
    
    Args:
        hidden_size (int): Size of hidden layers in MLP
        dropout_ratio (float): Dropout probability
        num_classes (int): Number of emotion classes
        feature_dim (int): Dimension of SSL model features
        ssl_model: Pre-trained SSL model (e.g., HuBERT, Wav2Vec2)
        device (str): Device to run the model on
    """
    
    def __init__(self, hidden_size, dropout_ratio, num_classes, feature_dim,
                 ssl_model=None, device='cuda'):
        super(SSLMLPModel, self).__init__()
        
        self.device = device
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        self.ssl_model = ssl_model
        self.feature_dim = feature_dim
        
        # Define MLP classifier
        self.dense = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_size, num_classes),
        ).to(device)
    
    def forward(self, x):
        """Forward pass through SSL model and MLP classifier."""
        # Extract features using SSL model
        x = self.ssl_model(x).last_hidden_state
        x = x.mean(dim=1)  # Global average pooling
        
        # Forward pass through MLP
        x = self.dense(x)
        return x


class RAVDESSDataset(Dataset):
    """
    PyTorch Dataset for RAVDESS emotion recognition dataset.
    
    Args:
        file_paths (list): List of audio file paths
        labels (list): List of corresponding emotion labels
        feature_extractor: HuggingFace feature extractor
        sample_rate (int): Target sample rate for audio
        max_audio_duration (float): Maximum audio duration in seconds
    """
    
    def __init__(self, file_paths, labels, feature_extractor=None,
                 sample_rate=16000, max_audio_duration=10.0):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.max_audio_duration = max_audio_duration
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load audio file
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Use random file as fallback
            random_idx = random.randint(0, len(self.file_paths) - 1)
            random_path = self.file_paths[random_idx]
            waveform, sr = torchaudio.load(random_path)
            print(f"Using fallback: {random_path}")
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract features
        inputs = self.feature_extractor(
            waveform.squeeze(),
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            max_length=int(self.feature_extractor.sampling_rate * self.max_audio_duration),
            truncation=True,
            padding='max_length'
        )
        
        return inputs['input_values'].squeeze(), label


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_ravdess_filename(filename):
    """Parse RAVDESS filename to extract metadata."""
    parts = os.path.basename(filename).split('.')[0].split('-')
    
    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    
    return {
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion': emotion_map.get(parts[2], 'unknown'),
        'emotion_id': parts[2],
        'intensity': 'normal' if parts[3] == '01' else 'strong',
        'actor_id': parts[6],
        'gender': 'male' if int(parts[6]) % 2 == 1 else 'female'
    }


def load_ravdess_data(root_path):
    """
    Load RAVDESS dataset with normal intensity only.
    
    Args:
        root_path (str): Root directory of RAVDESS dataset
        
    Returns:
        tuple: (file_paths, labels, num_classes)
    """
    emotion_map = {
        'sad': 0, 'calm': 1, 'surprised': 2, 'disgust': 3,
        'happy': 4, 'fearful': 5, 'neutral': 6, 'angry': 7
    }
    
    file_paths = []
    labels = []
    
    # Find all actor directories
    actor_dirs = glob.glob(os.path.join(root_path, 'Actor_*'))
    
    for actor_dir in actor_dirs:
        wav_files = glob.glob(os.path.join(actor_dir, '*.wav'))
        
        for wav_file in wav_files:
            info = parse_ravdess_filename(wav_file)
            
            # Only include normal intensity, audio-only speech files
            if (info['modality'] == '03' and 
                info['vocal_channel'] == '01' and 
                info['intensity'] == 'normal' and
                info['emotion'] in emotion_map):
                
                file_paths.append(wav_file)
                labels.append(emotion_map[info['emotion']])
    
    return file_paths, labels, len(emotion_map)


def load_pretrained_model(model_path, ssl_model_name, hidden_size, 
                         num_classes, feature_dim=768, device='cuda'):
    """Load pre-trained model from file."""
    ssl_model = AutoModel.from_pretrained(ssl_model_name)
    
    model = SSLMLPModel(
        hidden_size=hidden_size,
        dropout_ratio=0.1,
        feature_dim=feature_dim,
        num_classes=num_classes,
        ssl_model=ssl_model,
        device=device
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.eval()


def evaluate_model(model, data_loader, criterion, device='cuda'):
    """Evaluate model on given data loader."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return total_loss / len(data_loader), acc, f1, all_preds, all_labels


def print_results_header():
    """Print formatted header for results."""
    print("\n" + "="*80)
    print("RAVDESS EMOTION RECOGNITION EVALUATION RESULTS")
    print("="*80)


def print_fold_results(fold, acc, f1):
    """Print results for a single fold."""
    print(f"│ Fold {fold:1d} │ {acc:8.4f} │ {f1:8.4f} │")


def print_summary_results(results):
    """Print formatted summary of cross-validation results."""
    print("├────────┼──────────┼──────────┤")
    print(f"│ Mean   │ {results['accuracy_mean']:8.4f} │ {results['f1_mean']:8.4f} │")
    print(f"│ Std    │ {results['accuracy_std']:8.4f} │ {results['f1_std']:8.4f} │")
    print("└────────┴──────────┴──────────┘")


def cross_validation(file_paths, labels, feature_extractor, model_path, 
                    ssl_model_name, hidden_size, feature_dim, batch_size, 
                    num_classes, n_splits=5, seed=42):
    """Perform k-fold cross-validation."""
    set_seed(seed)
    
    # Combine data for splitting
    data = list(zip(file_paths, labels))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Initialize metrics storage
    accuracies = []
    f1_scores = []
    all_predictions = []
    all_true_labels = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print results header
    print("\n┌────────┬──────────┬──────────┐")
    print("│  Fold  │ Accuracy │ F1 Score │")
    print("├────────┼──────────┼──────────┤")
    
    # Run cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        # Split data
        test_data = [data[i] for i in test_idx]
        test_file_paths, test_labels = zip(*test_data)
        
        # Create test dataset and loader
        test_dataset = RAVDESSDataset(
            file_paths=test_file_paths,
            labels=test_labels,
            feature_extractor=feature_extractor,
            sample_rate=16000,
            max_audio_duration=5.0
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Load and evaluate model
        model = load_pretrained_model(
            model_path=model_path,
            ssl_model_name=ssl_model_name,
            hidden_size=hidden_size,
            num_classes=num_classes,
            feature_dim=feature_dim,
            device=device
        )
        
        criterion = nn.CrossEntropyLoss()
        _, acc, f1, preds, true_labels = evaluate_model(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device
        )
        
        # Store results
        accuracies.append(acc)
        f1_scores.append(f1)
        all_predictions.extend(preds)
        all_true_labels.extend(true_labels)
        
        # Print fold results
        print_fold_results(fold + 1, acc, f1)
    
    # Calculate summary statistics
    results = {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'fold_accuracies': accuracies,
        'fold_f1_scores': f1_scores
    }
    
    # Print summary
    print_summary_results(results)
    
    return results, all_predictions, all_true_labels


def save_results(results, predictions, true_labels, args):
    """Save evaluation results to JSON file."""
    # Create comprehensive results dictionary
    output_data = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'model_path': args.model_path,
            'ssl_model_name': args.ssl_model_name,
            'dataset_root': args.ravdess_root,
            'hidden_size': args.hidden_size,
            'feature_dim': args.feature_dim,
            'batch_size': args.batch_size,
            'seed': args.seed
        },
        'cross_validation_results': results,
        'emotion_mapping': {
            0: 'sad', 1: 'calm', 2: 'surprised', 3: 'disgust',
            4: 'happy', 5: 'fearful', 6: 'neutral', 7: 'angry'
        }
    }
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.ssl_model_name.split('/')[-1]
    output_path = f"ravdess_evaluation_{model_name}_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return output_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained emotion recognition models on RAVDESS dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the pre-trained model file (.pth)"
    )
    parser.add_argument(
        "--ravdess_root", type=str, required=True,
        help="Root directory of the RAVDESS dataset"
    )
    parser.add_argument(
        "--ssl_model_name", type=str, default="facebook/hubert-base-ls960",
        help="SSL model name used in training"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=128,
        help="Hidden size used in the model"
    )
    parser.add_argument(
        "--feature_dim", type=int, default=768,
        help="Feature dimension of the SSL model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    print_results_header()
    
    # Parse arguments
    args = parse_args()
    
    # Set seed and device
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Configuration:")
    print(f"  Model Path: {args.model_path}")
    print(f"  SSL Model: {args.ssl_model_name}")
    print(f"  Dataset: {args.ravdess_root}")
    print(f"  Device: {device}")
    print(f"  Random Seed: {args.seed}")
    
    # Load feature extractor
    print(f"\nLoading feature extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.ssl_model_name)
    
    # Load RAVDESS data (normal intensity only)
    print(f"Loading RAVDESS data...")
    file_paths, labels, num_classes = load_ravdess_data(args.ravdess_root)
    print(f"  Total samples: {len(file_paths)}")
    print(f"  Number of classes: {num_classes}")
    
    # Perform cross-validation
    print(f"\nPerforming 5-fold cross-validation...")
    results, predictions, true_labels = cross_validation(
        file_paths=file_paths,
        labels=labels,
        feature_extractor=feature_extractor,
        model_path=args.model_path,
        ssl_model_name=args.ssl_model_name,
        hidden_size=args.hidden_size,
        feature_dim=args.feature_dim,
        batch_size=args.batch_size,
        num_classes=num_classes,
        n_splits=5,
        seed=args.seed
    )
    
    # Save results
    output_path = save_results(results, predictions, true_labels, args)
    
    print(f"\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Final Results:")
    print(f"  Average Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"  Average F1 Score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    print(f"  Results saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()