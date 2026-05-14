import os
import yaml
import torch
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report
)
from scipy import stats
import warnings
import re
import json

warnings.filterwarnings('ignore')

from src.data_loader import create_data_loader
from src.preprocessing import load_dataset
from src.utils import calculate_metrics, plot_roc_curve, plot_pr_curve, plot_confusion_matrix, PROPERTY_NAMES
from models.multimodal_cnn import AMPPredictionModel
from src.utils import progress_bar, time_estimator, Tokenizer


def fix_config_file(config_path):
    """Fix configuration file, remove Python objects"""
    print(f"Fixing configuration file: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if Python objects exist
    if '!!python/object:src.utils.Tokenizer' in content:
        print("Python objects detected, fixing...")

        # Use default111.yaml as template
        default_config_path = "configs/default.yaml"
        if os.path.exists(default_config_path):
            with open(default_config_path, 'r', encoding='utf-8') as f:
                default_config = yaml.safe_load(f)

            # Read current configuration but skip tokenizer
            with open(config_path, 'r', encoding='utf-8') as f:
                current_config = yaml.load(f, Loader=yaml.FullLoader)

            # Merge configurations, remove tokenizer object
            if 'data' in current_config and 'tokenizer' in current_config['data']:
                del current_config['data']['tokenizer']

            # Ensure necessary paths exist
            if 'data' in current_config:
                # Use default paths from default111.yaml
                current_config['data']['train_path'] = default_config['data']['train_path']
                current_config['data']['val_path'] = default_config['data']['val_path']
                current_config['data']['test_path'] = default_config['data']['test_path']

            # Save fixed configuration
            backup_path = config_path + '.backup'
            os.rename(config_path, backup_path)
            print(f"Original configuration backed up to: {backup_path}")

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(current_config, f, default_flow_style=False)

            print(f"✅ Configuration file fixed: {config_path}")
            return True
        else:
            print(f"⚠️ Warning: Default configuration file not found: {default_config_path}")
            return False
    else:
        print("✅ Configuration file does not need fixing")
        return True


class ModelEvaluator:
    def __init__(self, checkpoints_dir="./checkpoints", output_dir="./evaluation_results"):
        self.checkpoints_dir = checkpoints_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Fix configuration file
        config_path = os.path.join(checkpoints_dir, "config.yaml")
        fix_config_file(config_path)

        # Load configuration file
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Add tokenizer
        self.config['data']['tokenizer'] = Tokenizer()

        # Get all model files
        self.model_files = self._get_model_files()
        print(f"Found {len(self.model_files)} model files:")
        for model_file in self.model_files:
            print(f"  - {model_file}")

        # Load test data
        self.test_sequences, self.test_labels, self.test_properties = self._load_test_data()

        # Device settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Store all model predictions for voting
        self.all_model_predictions = {}

        # Store all raw data for plotting
        self.raw_data = {
            'sequence_data': [],
            'model_predictions': {},
            'voting_data': {},
            'curve_data': {}
        }

    def _get_model_files(self):
        """Get all model files, exclude configuration files"""
        model_files = []
        for file in os.listdir(self.checkpoints_dir):
            if file.endswith(".pth"):
                model_files.append(os.path.join(self.checkpoints_dir, file))

        # Sort by F1 score
        def extract_f1(filename):
            match = re.search(r'f1_([\d.]+)\.pth', filename)
            if match:
                return float(match.group(1))
            return 0.0

        model_files.sort(key=extract_f1, reverse=True)
        return model_files

    def _load_test_data(self):
        """Load test data"""
        test_data_path = self.config['data']['test_path']
        print(f"Loading test data: {test_data_path}")

        # If test path doesn't exist, try other paths
        if not os.path.exists(test_data_path):
            print(f"⚠️ Warning: Test path not found: {test_data_path}")
            # Try to find other possible test files
            possible_paths = [
                "data/processed_new/test.csv",
                "data/processed/test.csv",
                "./test.csv"
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    test_data_path = path
                    print(f"Using test data: {test_data_path}")
                    break

        test_sequences, test_labels, test_properties = load_dataset(
            test_data_path,
            max_len=self.config['data']['max_seq_len']
        )

        print(f"Test data statistics:")
        print(f"  Number of sequences: {len(test_sequences)}")
        print(f"  Positive samples: {np.sum(test_labels == 1)}")
        print(f"  Negative samples: {np.sum(test_labels == 0)}")

        return test_sequences, test_labels, test_properties

    def _create_test_loader(self):
        """Create test data loader"""
        test_loader = create_data_loader(
            sequences=self.test_sequences,
            labels=self.test_labels,
            properties=self.test_properties,
            config=self.config,
            shuffle=False
        )
        return test_loader

    def _detect_model_property_count(self, model_path):
        """Detect the number of properties used in a model from its state_dict"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)

            # Look for property-related weights in the state_dict
            for key in state_dict.keys():
                if 'property_importance_module' in key and 'weight' in key:
                    weight_shape = state_dict[key].shape
                    # The second dimension of the weight is the number of input features
                    if len(weight_shape) == 2:
                        return weight_shape[1]
                    elif len(weight_shape) == 3:
                        # For conv layers: [out_channels, in_channels, kernel_size]
                        return weight_shape[1]

            # If not found, check property_projection layer
            for key in state_dict.keys():
                if 'property_projection' in key and 'weight' in key:
                    weight_shape = state_dict[key].shape
                    if len(weight_shape) == 2:
                        return weight_shape[1]

            return None
        except Exception as e:
            print(f"Error detecting property count: {e}")
            return None

    def _get_property_names_for_count(self, count):
        """Get property names for a given count"""
        if count <= len(PROPERTY_NAMES):
            return PROPERTY_NAMES[:count]
        else:
            # If count is larger than available properties, use all and add dummy ones
            property_names = list(PROPERTY_NAMES)
            for i in range(count - len(PROPERTY_NAMES)):
                property_names.append(f"dummy_property_{i + 1}")
            return property_names

    def evaluate_single_model(self, model_path, model_name="model"):
        """Evaluate a single model"""
        print(f"\n{'=' * 60}")
        print(f"Evaluating model: {os.path.basename(model_path)}")
        print(f"{'=' * 60}")

        # Try to detect the number of properties used in this model
        property_count = self._detect_model_property_count(model_path)

        if property_count is None:
            print("⚠️ Warning: Could not detect property count, using default (4)")
            property_count = 4

        print(f"Detected property count for model: {property_count}")

        # Update config with detected property count
        model_config = self.config.copy()
        if 'model' not in model_config:
            model_config['model'] = {}

        # Use the appropriate number of properties
        property_names = self._get_property_names_for_count(property_count)
        model_config['model']['important_properties'] = property_names

        # Create model
        try:
            model = AMPPredictionModel(model_config['model']).to(self.device)

            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)

            # Filter state_dict to remove incompatible keys
            model_state_dict = model.state_dict()
            filtered_state_dict = {}

            for key, value in state_dict.items():
                if key in model_state_dict:
                    if model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        print(f"⚠️ Shape mismatch for {key}: {value.shape} vs {model_state_dict[key].shape}")
                else:
                    print(f"⚠️ Key not found in model: {key}")

            # Load filtered state_dict
            model.load_state_dict(filtered_state_dict, strict=False)
            print("✅ Model weights loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model weights: {e}")
            return None

        model.eval()

        # Create data loader
        test_loader = self._create_test_loader()

        # Perform evaluation
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch_data in progress_bar(test_loader, desc=f"Evaluating {model_name}", unit="batch"):
                seq = batch_data['seq_ids'].to(self.device)
                labels = batch_data['labels'].to(self.device)
                props = batch_data['properties'].to(self.device)
                orig_sequences = batch_data['sequence_strs']

                outputs = model(seq, props, orig_sequences)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)

                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(preds)
                all_probs.extend(probs)

        # Calculate metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        metrics = self._calculate_detailed_metrics(all_labels, all_preds, all_probs)

        # Store predictions for voting
        self.all_model_predictions[model_name] = {
            'probs': all_probs,
            'preds': all_preds,
            'metrics': metrics,
            'model_file': os.path.basename(model_path),
            'property_count': property_count
        }

        # Store raw data for plotting
        self.raw_data['model_predictions'][model_name] = {
            'probabilities': all_probs.tolist(),
            'predictions': all_preds.tolist(),
            'roc_curve': {
                'fpr': metrics['fpr'].tolist() if isinstance(metrics['fpr'], np.ndarray) else metrics['fpr'],
                'tpr': metrics['tpr'].tolist() if isinstance(metrics['tpr'], np.ndarray) else metrics['tpr']
            },
            'pr_curve': {
                'precision': metrics['precision_curve'].tolist() if isinstance(metrics['precision_curve'],
                                                                               np.ndarray) else metrics[
                    'precision_curve'],
                'recall': metrics['recall_curve'].tolist() if isinstance(metrics['recall_curve'], np.ndarray) else
                metrics['recall_curve']
            }
        }

        # Save results
        self._save_single_model_results(model_name, os.path.basename(model_path),
                                        all_labels, all_preds, all_probs, metrics)

        return metrics

    def _calculate_detailed_metrics(self, y_true, y_pred, y_prob):
        """Calculate detailed metrics with 4 decimal precision"""
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # ROC AUC
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
        else:
            metrics['roc_auc'] = 0.5
            metrics['fpr'] = np.array([0, 1])
            metrics['tpr'] = np.array([0, 1])

        # PR AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        metrics['pr_auc'] = auc(recall_curve, precision_curve)
        metrics['precision_curve'] = precision_curve
        metrics['recall_curve'] = recall_curve

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['mcc'] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (
                                                                                                                         tp + fp) * (
                                                                                                                         tp + fn) * (
                                                                                                                         tn + fp) * (
                                                                                                                         tn + fn) > 0 else 0
        else:
            metrics['specificity'] = 0
            metrics['mcc'] = 0

        return metrics

    def _save_single_model_results(self, model_name, model_file, y_true, y_pred, y_prob, metrics):
        """Save single model evaluation results"""
        # Create model-specific directory
        model_dir = os.path.join(self.output_dir, f"single_model_{model_name}")
        os.makedirs(model_dir, exist_ok=True)

        # Save detailed prediction results
        results_df = pd.DataFrame({
            'sequence': self.test_sequences,
            'true_label': y_true,
            'predicted_label': y_pred,
            'probability': y_prob,
            'correct': (y_true == y_pred).astype(int)
        })
        results_path = os.path.join(model_dir, "detailed_predictions.csv")
        results_df.to_csv(results_path, index=False)
        print(f"📊 Detailed prediction results saved: {results_path}")

        # Save metrics
        metrics_path = os.path.join(model_dir, "metrics.csv")
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(metrics_path, index=False)

        # Save metrics as text file
        metrics_txt_path = os.path.join(model_dir, "metrics.txt")
        with open(metrics_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Model Evaluation Results: {model_file}\n")
            f.write(f"Evaluation Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"  Specificity: {metrics['specificity']:.4f}\n")
            f.write(f"  MCC: {metrics['mcc']:.4f}\n")
            f.write(f"  ROC AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"  PR AUC: {metrics['pr_auc']:.4f}\n")

        # Generate visualization charts
        self._generate_visualizations(model_dir, model_name, y_true, y_pred, y_prob, metrics)

        print(f"✅ Model evaluation completed, results saved in: {model_dir}")

    def _generate_visualizations(self, save_dir, model_name, y_true, y_pred, y_prob, metrics):
        """Generate visualization charts in English"""
        plt.style.use('seaborn-v0_8-whitegrid')

        # 1. ROC Curve
        plt.figure(figsize=(10, 8))
        if 'fpr' in metrics and 'tpr' in metrics:
            plt.plot(metrics['fpr'], metrics['tpr'],
                     label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})',
                     linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Guessing')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(save_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()

        # 2. PR Curve
        plt.figure(figsize=(10, 8))
        if 'recall_curve' in metrics and 'precision_curve' in metrics:
            plt.plot(metrics['recall_curve'], metrics['precision_curve'],
                     label=f'PR Curve (AUC = {metrics["pr_auc"]:.4f})',
                     linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'PR Curve - {model_name}', fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pr_path = os.path.join(save_dir, "pr_curve.png")
        plt.savefig(pr_path, dpi=150)
        plt.close()

        # 3. Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()

        # 4. Probability Distribution Plot
        plt.figure(figsize=(12, 8))
        for label in [0, 1]:
            mask = y_true == label
            if np.any(mask):
                plt.hist(y_prob[mask], bins=30, alpha=0.6,
                         label=f'True Label={label} (n={np.sum(mask)})',
                         density=True)
        plt.xlabel('Prediction Probability', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'Prediction Probability Distribution - {model_name}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        prob_path = os.path.join(save_dir, "probability_distribution.png")
        plt.savefig(prob_path, dpi=150)
        plt.close()

        print(f"📈 Visualization charts saved to: {save_dir}")

    def evaluate_all_models(self):
        """Evaluate all models"""
        print(f"\n{'=' * 80}")
        print(f"Starting evaluation of {len(self.model_files)} models")
        print(f"{'=' * 80}")

        all_metrics = []
        all_predictions = []

        # Store sequence data
        self.raw_data['sequence_data'] = {
            'sequences': self.test_sequences.tolist() if isinstance(self.test_sequences,
                                                                    np.ndarray) else self.test_sequences,
            'true_labels': self.test_labels.tolist() if isinstance(self.test_labels, np.ndarray) else self.test_labels
        }

        # Evaluate each model
        for i, model_path in enumerate(self.model_files, 1):
            model_name = f"Model_{i:02d}"
            print(f"\nProgress: {i}/{len(self.model_files)}")

            metrics = self.evaluate_single_model(model_path, model_name)
            if metrics is not None:
                # Add model information
                metrics['model_name'] = model_name
                metrics['model_file'] = os.path.basename(model_path)
                metrics['model_index'] = i

                # Extract F1 score (for sorting)
                metrics['f1_from_filename'] = self._extract_f1_from_filename(model_path)

                all_metrics.append(metrics)

                # Load prediction results
                model_dir = os.path.join(self.output_dir, f"single_model_{model_name}")
                pred_path = os.path.join(model_dir, "detailed_predictions.csv")
                if os.path.exists(pred_path):
                    pred_df = pd.read_csv(pred_path)
                    pred_df['model_name'] = model_name
                    all_predictions.append(pred_df)

        # Save summary of all model results
        self._save_all_models_summary(all_metrics)

        # Perform voting and ensemble evaluation
        if len(all_metrics) >= 2:
            # Ensemble evaluation
            self.evaluate_ensemble(all_metrics, all_predictions)

            # Voting evaluation
            if len(self.all_model_predictions) > 0:
                self.evaluate_voting(all_metrics)

        print(f"\n{'=' * 80}")
        print(f"All model evaluations completed!")
        print(f"Results saved in: {self.output_dir}")
        print(f"{'=' * 80}")

    def _extract_f1_from_filename(self, filename):
        """Extract F1 score from filename"""
        import re
        match = re.search(r'f1_([\d.]+)\.pth', os.path.basename(filename))
        if match:
            return float(match.group(1))
        return 0.0

    def evaluate_voting(self, all_metrics):
        """Evaluate voting ensemble of all models"""
        print(f"\n{'=' * 80}")
        print(f"Starting Voting Ensemble Evaluation")
        print(f"{'=' * 80}")

        # Create voting evaluation directory
        voting_dir = os.path.join(self.output_dir, "voting_evaluation")
        os.makedirs(voting_dir, exist_ok=True)

        if len(self.all_model_predictions) == 0:
            print("❌ No model predictions available for voting")
            return

        print(f"Number of models for voting: {len(self.all_model_predictions)}")

        # Get true labels from first model
        first_model_name = list(self.all_model_predictions.keys())[0]
        first_model_dir = os.path.join(self.output_dir, f"single_model_{first_model_name}")
        first_model_pred_path = os.path.join(first_model_dir, "detailed_predictions.csv")

        if not os.path.exists(first_model_pred_path):
            print("❌ Cannot find prediction results for voting")
            return

        # Load true labels
        pred_df = pd.read_csv(first_model_pred_path)
        y_true = pred_df['true_label'].values

        # Collect predictions from all models
        all_probs = []
        all_preds = []
        model_names = []

        for model_name, predictions in self.all_model_predictions.items():
            all_probs.append(predictions['probs'])
            all_preds.append(predictions['preds'])
            model_names.append(model_name)

        # Convert to arrays
        all_probs = np.array(all_probs)  # Shape: (n_models, n_samples)
        all_preds = np.array(all_preds)  # Shape: (n_models, n_samples)

        print(f"All predictions shape: {all_preds.shape}")
        print(f"All probabilities shape: {all_probs.shape}")

        # Calculate voting results
        # 1. Hard voting: majority vote
        voting_preds_hard = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)

        # 2. Soft voting: average probability > 0.5
        avg_probs = np.mean(all_probs, axis=0)
        voting_preds_soft = (avg_probs > 0.5).astype(int)

        # 3. Weighted voting: weight by model F1 score
        f1_scores = [pred['metrics']['f1'] for pred in self.all_model_predictions.values()]
        weights = np.array(f1_scores) / np.sum(f1_scores)
        weighted_avg_probs = np.average(all_probs, axis=0, weights=weights)
        voting_preds_weighted = (weighted_avg_probs > 0.5).astype(int)

        # Calculate metrics for each voting method
        voting_metrics = {}

        # Hard voting metrics
        hard_metrics = self._calculate_detailed_metrics(y_true, voting_preds_hard, avg_probs)
        voting_metrics['hard_voting'] = {
            'metrics': hard_metrics,
            'predictions': voting_preds_hard,
            'probabilities': avg_probs
        }

        # Soft voting metrics
        soft_metrics = self._calculate_detailed_metrics(y_true, voting_preds_soft, avg_probs)
        voting_metrics['soft_voting'] = {
            'metrics': soft_metrics,
            'predictions': voting_preds_soft,
            'probabilities': avg_probs
        }

        # Weighted voting metrics
        weighted_metrics = self._calculate_detailed_metrics(y_true, voting_preds_weighted, weighted_avg_probs)
        voting_metrics['weighted_voting'] = {
            'metrics': weighted_metrics,
            'predictions': voting_preds_weighted,
            'probabilities': weighted_avg_probs
        }

        # Save voting results
        voting_results_df = pd.DataFrame({
            'sequence': pred_df['sequence'],
            'true_label': y_true,
            'hard_voting_pred': voting_preds_hard,
            'soft_voting_pred': voting_preds_soft,
            'weighted_voting_pred': voting_preds_weighted,
            'average_probability': avg_probs,
            'weighted_probability': weighted_avg_probs
        })

        # Add individual model predictions to DataFrame
        for i, model_name in enumerate(model_names):
            voting_results_df[f'{model_name}_prob'] = all_probs[i]
            voting_results_df[f'{model_name}_pred'] = all_preds[i]

        voting_results_path = os.path.join(voting_dir, "voting_predictions_detailed.csv")
        voting_results_df.to_csv(voting_results_path, index=False)
        print(f"📊 Detailed voting predictions saved: {voting_results_path}")

        # Save voting metrics
        voting_metrics_path = os.path.join(voting_dir, "voting_metrics.txt")
        with open(voting_metrics_path, 'w', encoding='utf-8') as f:
            f.write("Voting Ensemble Evaluation Results\n")
            f.write("=" * 80 + "\n")
            f.write(f"Number of models: {len(self.all_model_predictions)}\n")
            f.write(f"Evaluation Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write("Hard Voting (Majority Vote):\n")
            f.write(f"  Accuracy: {hard_metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {hard_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {hard_metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {hard_metrics['f1']:.4f}\n")
            f.write(f"  ROC AUC: {hard_metrics['roc_auc']:.4f}\n")
            f.write(f"  PR AUC: {hard_metrics['pr_auc']:.4f}\n\n")

            f.write("Soft Voting (Average Probability > 0.5):\n")
            f.write(f"  Accuracy: {soft_metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {soft_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {soft_metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {soft_metrics['f1']:.4f}\n")
            f.write(f"  ROC AUC: {soft_metrics['roc_auc']:.4f}\n")
            f.write(f"  PR AUC: {soft_metrics['pr_auc']:.4f}\n\n")

            f.write("Weighted Voting (Weighted by F1 Score):\n")
            f.write(f"  Accuracy: {weighted_metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {weighted_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {weighted_metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {weighted_metrics['f1']:.4f}\n")
            f.write(f"  ROC AUC: {weighted_metrics['roc_auc']:.4f}\n")
            f.write(f"  PR AUC: {weighted_metrics['pr_auc']:.4f}\n\n")

            # Model weights
            f.write("Model Weights (for weighted voting):\n")
            for i, (model_name, weight) in enumerate(zip(model_names, weights)):
                f.write(f"  {model_name}: {weight:.4f} (F1: {f1_scores[i]:.4f})\n")

        # Store raw voting data
        self.raw_data['voting_data'] = {
            'true_labels': y_true.tolist(),
            'model_names': model_names,
            'model_predictions': all_preds.tolist(),
            'model_probabilities': all_probs.tolist(),
            'hard_voting': {
                'predictions': voting_preds_hard.tolist(),
                'probabilities': avg_probs.tolist(),
                'metrics': hard_metrics
            },
            'soft_voting': {
                'predictions': voting_preds_soft.tolist(),
                'probabilities': avg_probs.tolist(),
                'metrics': soft_metrics
            },
            'weighted_voting': {
                'predictions': voting_preds_weighted.tolist(),
                'probabilities': weighted_avg_probs.tolist(),
                'metrics': weighted_metrics
            },
            'model_weights': weights.tolist(),
            'model_f1_scores': f1_scores
        }

        # Generate comprehensive visualization with all models and voting methods
        self._generate_voting_visualizations(voting_dir, voting_metrics, all_metrics, y_true)

        # Save all raw data for plotting
        self._save_all_raw_data(voting_dir)

        print(f"\n✅ Voting ensemble evaluation completed!")
        print(f"   Number of models: {len(self.all_model_predictions)}")
        print(f"   Hard Voting F1: {hard_metrics['f1']:.4f}")
        print(f"   Soft Voting F1: {soft_metrics['f1']:.4f}")
        print(f"   Weighted Voting F1: {weighted_metrics['f1']:.4f}")
        print(f"   Results saved in: {voting_dir}")

    def _save_all_raw_data(self, save_dir):
        """Save all raw data for plotting"""
        print("\n💾 Saving all raw data for plotting...")

        # 1. Save raw data as JSON (for easy loading)
        json_path = os.path.join(save_dir, "raw_plotting_data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = self._make_data_serializable(self.raw_data)
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        print(f"📁 Raw data JSON saved: {json_path}")

        # 2. Save detailed ROC curve data
        roc_data_dir = os.path.join(save_dir, "roc_curve_data")
        os.makedirs(roc_data_dir, exist_ok=True)

        # Save individual model ROC data
        for model_name, model_data in self.raw_data['model_predictions'].items():
            roc_df = pd.DataFrame({
                'fpr': model_data['roc_curve']['fpr'],
                'tpr': model_data['roc_curve']['tpr']
            })
            roc_path = os.path.join(roc_data_dir, f"{model_name}_roc_curve.csv")
            roc_df.to_csv(roc_path, index=False)

        # Save voting methods ROC data
        if 'voting_data' in self.raw_data and self.raw_data['voting_data']:
            y_true = self.raw_data['voting_data']['true_labels']

            for method_name in ['hard_voting', 'soft_voting', 'weighted_voting']:
                if method_name in self.raw_data['voting_data']:
                    method_data = self.raw_data['voting_data'][method_name]
                    probs = np.array(method_data['probabilities'])
                    fpr, tpr, _ = roc_curve(y_true, probs)

                    roc_df = pd.DataFrame({
                        'fpr': fpr,
                        'tpr': tpr
                    })
                    roc_path = os.path.join(roc_data_dir, f"{method_name}_roc_curve.csv")
                    roc_df.to_csv(roc_path, index=False)

        print(f"📊 ROC curve data saved to: {roc_data_dir}")

        # 3. Save detailed PR curve data
        pr_data_dir = os.path.join(save_dir, "pr_curve_data")
        os.makedirs(pr_data_dir, exist_ok=True)

        # Save individual model PR data
        for model_name, model_data in self.raw_data['model_predictions'].items():
            pr_df = pd.DataFrame({
                'recall': model_data['pr_curve']['recall'],
                'precision': model_data['pr_curve']['precision']
            })
            pr_path = os.path.join(pr_data_dir, f"{model_name}_pr_curve.csv")
            pr_df.to_csv(pr_path, index=False)

        # Save voting methods PR data
        if 'voting_data' in self.raw_data and self.raw_data['voting_data']:
            y_true = self.raw_data['voting_data']['true_labels']

            for method_name in ['hard_voting', 'soft_voting', 'weighted_voting']:
                if method_name in self.raw_data['voting_data']:
                    method_data = self.raw_data['voting_data'][method_name]
                    probs = np.array(method_data['probabilities'])
                    precision, recall, _ = precision_recall_curve(y_true, probs)

                    pr_df = pd.DataFrame({
                        'recall': recall,
                        'precision': precision
                    })
                    pr_path = os.path.join(pr_data_dir, f"{method_name}_pr_curve.csv")
                    pr_df.to_csv(pr_path, index=False)

        print(f"📊 PR curve data saved to: {pr_data_dir}")

        # 4. Save model comparison data
        model_stats = []
        for model_name, model_data in self.all_model_predictions.items():
            model_stats.append({
                'model_name': model_name,
                'model_file': model_data['model_file'],
                'f1_score': model_data['metrics']['f1'],
                'accuracy': model_data['metrics']['accuracy'],
                'precision': model_data['metrics']['precision'],
                'recall': model_data['metrics']['recall'],
                'roc_auc': model_data['metrics']['roc_auc'],
                'pr_auc': model_data['metrics']['pr_auc'],
                'property_count': model_data['property_count']
            })

        model_stats_df = pd.DataFrame(model_stats)
        model_stats_path = os.path.join(save_dir, "model_statistics.csv")
        model_stats_df.to_csv(model_stats_path, index=False)
        print(f"📈 Model statistics saved: {model_stats_path}")

        # 5. Save comprehensive Excel file
        excel_path = os.path.join(save_dir, "all_plotting_data.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Model statistics
            model_stats_df.to_excel(writer, sheet_name='Model Statistics', index=False)

            # Voting results
            if 'voting_data' in self.raw_data and self.raw_data['voting_data']:
                voting_df = pd.DataFrame({
                    'sequence': self.raw_data['sequence_data']['sequences'],
                    'true_label': self.raw_data['voting_data']['true_labels']
                })

                # Add model predictions
                for i, model_name in enumerate(self.raw_data['voting_data']['model_names']):
                    voting_df[f'{model_name}_prob'] = self.raw_data['voting_data']['model_probabilities'][i]
                    voting_df[f'{model_name}_pred'] = self.raw_data['voting_data']['model_predictions'][i]

                # Add voting results
                for method_name in ['hard_voting', 'soft_voting', 'weighted_voting']:
                    if method_name in self.raw_data['voting_data']:
                        method_data = self.raw_data['voting_data'][method_name]
                        voting_df[f'{method_name}_pred'] = method_data['predictions']
                        voting_df[f'{method_name}_prob'] = method_data['probabilities']

                voting_df.to_excel(writer, sheet_name='Voting Results', index=False)

            # Model weights
            weights_df = pd.DataFrame({
                'model_name': self.raw_data['voting_data']['model_names'],
                'weight': self.raw_data['voting_data']['model_weights'],
                'f1_score': self.raw_data['voting_data']['model_f1_scores']
            })
            weights_df.to_excel(writer, sheet_name='Model Weights', index=False)

        print(f"📘 Comprehensive Excel file saved: {excel_path}")

    def _make_data_serializable(self, data):
        """Convert data to JSON serializable format"""
        if isinstance(data, dict):
            return {k: self._make_data_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_data_serializable(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        else:
            return data

    def _generate_voting_visualizations(self, save_dir, voting_metrics, all_metrics, y_true):
        """Generate comprehensive visualizations for voting evaluation"""
        plt.style.use('seaborn-v0_8-whitegrid')

        # Prepare data for ROC and PR curves
        model_curves = []

        # Get model predictions
        for model_name, predictions in self.all_model_predictions.items():
            # Store model ROC and PR data
            model_curves.append({
                'name': model_name,
                'probs': predictions['probs'],
                'metrics': predictions['metrics']
            })

        if len(model_curves) == 0:
            print("⚠️ Warning: No model predictions available for visualization")
            return

        # 1. Comprehensive ROC Curve (All Models + Voting Methods)
        plt.figure(figsize=(16, 12))

        # Plot individual model ROC curves
        colors = plt.cm.rainbow(np.linspace(0, 1, len(model_curves)))
        for i, model_data in enumerate(model_curves):
            fpr, tpr, _ = roc_curve(y_true, model_data['probs'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.5, alpha=0.7, color=colors[i],
                     label=f"{model_data['name']} (AUC = {roc_auc:.4f})")

        # Plot voting method ROC curves
        voting_colors = {'hard_voting': 'black', 'soft_voting': 'red', 'weighted_voting': 'blue'}
        voting_line_styles = {'hard_voting': '--', 'soft_voting': '-.', 'weighted_voting': ':'}

        for method_name, method_data in voting_metrics.items():
            fpr, tpr, _ = roc_curve(y_true, method_data['probabilities'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2.5, color=voting_colors.get(method_name, 'gray'),
                     linestyle=voting_line_styles.get(method_name, '-'),
                     label=f"{method_name.replace('_', ' ').title()} (AUC = {roc_auc:.4f})")

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Guessing')
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curves: All Models vs Voting Methods', fontsize=16)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(save_dir, "comprehensive_roc_curves.png")
        plt.savefig(roc_path, dpi=200, bbox_inches='tight')
        plt.close()

        # Store ROC curve data
        self.raw_data['curve_data']['roc'] = {
            'model_curves': [
                {
                    'name': curve['name'],
                    'fpr': curve['metrics']['fpr'].tolist() if isinstance(curve['metrics']['fpr'], np.ndarray) else
                    curve['metrics']['fpr'],
                    'tpr': curve['metrics']['tpr'].tolist() if isinstance(curve['metrics']['tpr'], np.ndarray) else
                    curve['metrics']['tpr']
                }
                for curve in model_curves
            ],
            'voting_curves': {
                method_name: {
                    'fpr': roc_curve(y_true, method_data['probabilities'])[0].tolist(),
                    'tpr': roc_curve(y_true, method_data['probabilities'])[1].tolist()
                }
                for method_name, method_data in voting_metrics.items()
            }
        }

        # 2. Comprehensive PR Curve (All Models + Voting Methods)
        plt.figure(figsize=(16, 12))

        # Plot individual model PR curves
        for i, model_data in enumerate(model_curves):
            precision, recall, _ = precision_recall_curve(y_true, model_data['probs'])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=1.5, alpha=0.7, color=colors[i],
                     label=f"{model_data['name']} (AUC = {pr_auc:.4f})")

        # Plot voting method PR curves
        for method_name, method_data in voting_metrics.items():
            precision, recall, _ = precision_recall_curve(y_true, method_data['probabilities'])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2.5, color=voting_colors.get(method_name, 'gray'),
                     linestyle=voting_line_styles.get(method_name, '-'),
                     label=f"{method_name.replace('_', ' ').title()} (AUC = {pr_auc:.4f})")

        # Add baseline
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5,
                    label=f'Baseline (P = {baseline:.4f})')

        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curves: All Models vs Voting Methods', fontsize=16)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pr_path = os.path.join(save_dir, "comprehensive_pr_curves.png")
        plt.savefig(pr_path, dpi=200, bbox_inches='tight')
        plt.close()

        # Store PR curve data
        self.raw_data['curve_data']['pr'] = {
            'model_curves': [
                {
                    'name': curve['name'],
                    'recall': curve['metrics']['recall_curve'].tolist() if isinstance(curve['metrics']['recall_curve'],
                                                                                      np.ndarray) else curve['metrics'][
                        'recall_curve'],
                    'precision': curve['metrics']['precision_curve'].tolist() if isinstance(
                        curve['metrics']['precision_curve'], np.ndarray) else curve['metrics']['precision_curve']
                }
                for curve in model_curves
            ],
            'voting_curves': {
                method_name: {
                    'recall': precision_recall_curve(y_true, method_data['probabilities'])[1].tolist(),
                    'precision': precision_recall_curve(y_true, method_data['probabilities'])[0].tolist()
                }
                for method_name, method_data in voting_metrics.items()
            },
            'baseline': float(baseline)
        }

        # 3. Voting Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # F1 Score comparison
        ax1 = axes[0, 0]
        f1_scores = [
            voting_metrics['hard_voting']['metrics']['f1'],
            voting_metrics['soft_voting']['metrics']['f1'],
            voting_metrics['weighted_voting']['metrics']['f1']
        ]
        voting_methods = ['Hard Voting', 'Soft Voting', 'Weighted Voting']
        bar_colors = ['skyblue', 'lightgreen', 'lightcoral']

        bars = ax1.bar(voting_methods, f1_scores, color=bar_colors, alpha=0.7)
        ax1.set_ylabel('F1 Score', fontsize=12)
        ax1.set_title('Voting Methods F1 Score Comparison', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                     f'{score:.4f}', ha='center', va='bottom', fontsize=10)

        # ROC AUC comparison
        ax2 = axes[0, 1]
        roc_auc_scores = [
            voting_metrics['hard_voting']['metrics']['roc_auc'],
            voting_metrics['soft_voting']['metrics']['roc_auc'],
            voting_metrics['weighted_voting']['metrics']['roc_auc']
        ]

        bars = ax2.bar(voting_methods, roc_auc_scores, color=bar_colors, alpha=0.7)
        ax2.set_ylabel('ROC AUC', fontsize=12)
        ax2.set_title('Voting Methods ROC AUC Comparison', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, score in zip(bars, roc_auc_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                     f'{score:.4f}', ha='center', va='bottom', fontsize=10)

        # Accuracy comparison
        ax3 = axes[1, 0]
        accuracy_scores = [
            voting_metrics['hard_voting']['metrics']['accuracy'],
            voting_metrics['soft_voting']['metrics']['accuracy'],
            voting_metrics['weighted_voting']['metrics']['accuracy']
        ]

        bars = ax3.bar(voting_methods, accuracy_scores, color=bar_colors, alpha=0.7)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Voting Methods Accuracy Comparison', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='y')

        for bar, score in zip(bars, accuracy_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                     f'{score:.4f}', ha='center', va='bottom', fontsize=10)

        # Model weight distribution
        ax4 = axes[1, 1]
        if 'voting_data' in self.raw_data and 'model_weights' in self.raw_data['voting_data']:
            model_names_short = [name[-2:] for name in self.raw_data['voting_data']['model_names']]
            weights = self.raw_data['voting_data']['model_weights']

            bars = ax4.bar(range(len(model_names_short)), weights, color='lightblue', alpha=0.7)
            ax4.set_xlabel('Model', fontsize=12)
            ax4.set_ylabel('Weight', fontsize=12)
            ax4.set_title('Model Weights for Weighted Voting', fontsize=14)
            ax4.set_xticks(range(len(model_names_short)))
            ax4.set_xticklabels(model_names_short, rotation=45)
            ax4.grid(True, alpha=0.3, axis='y')

            # Add weight values on bars
            for bar, weight in zip(bars, weights):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                         f'{weight:.4f}', ha='center', va='bottom', fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'No model weight data available',
                     ha='center', va='center', fontsize=12, transform=ax4.transAxes)
            ax4.set_title('Model Weights for Weighted Voting', fontsize=14)
            ax4.set_xticks([])
            ax4.set_yticks([])

        plt.tight_layout()
        comparison_path = os.path.join(save_dir, "voting_performance_comparison.png")
        plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
        plt.close()

        # Store performance comparison data
        self.raw_data['curve_data']['performance_comparison'] = {
            'voting_methods': voting_methods,
            'f1_scores': f1_scores,
            'roc_auc_scores': roc_auc_scores,
            'accuracy_scores': accuracy_scores,
        }

        # Add model weights if available
        if 'voting_data' in self.raw_data and 'model_weights' in self.raw_data['voting_data']:
            self.raw_data['curve_data']['performance_comparison']['model_weights'] = {
                'model_names': [name[-2:] for name in self.raw_data['voting_data']['model_names']],
                'weights': self.raw_data['voting_data']['model_weights']
            }

        # 4. Confusion Matrices for Voting Methods
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (method_name, ax) in enumerate(zip(['hard_voting', 'soft_voting', 'weighted_voting'], axes)):
            cm = voting_metrics[method_name]['metrics']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Predicted Negative', 'Predicted Positive'],
                        yticklabels=['Actual Negative', 'Actual Positive'])
            ax.set_title(f'{method_name.replace("_", " ").title()} Confusion Matrix', fontsize=14)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=12)

        plt.tight_layout()
        cm_path = os.path.join(save_dir, "voting_confusion_matrices.png")
        plt.savefig(cm_path, dpi=200, bbox_inches='tight')
        plt.close()

        # 5. Probability Distribution for Voting Methods
        plt.figure(figsize=(14, 10))

        # Plot probability distributions for each voting method
        for method_name, method_data in voting_metrics.items():
            probs = method_data['probabilities']
            color = voting_colors.get(method_name, 'gray')
            plt.hist(probs, bins=30, alpha=0.5, density=True,
                     label=f"{method_name.replace('_', ' ').title()} (n={len(probs)})",
                     color=color)

        plt.xlabel('Average Prediction Probability', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('Probability Distribution for Voting Methods', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        prob_path = os.path.join(save_dir, "voting_probability_distributions.png")
        plt.savefig(prob_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"📈 Voting visualization charts saved to: {save_dir}")

    def _save_all_models_summary(self, all_metrics):
        """Save summary of all model results"""
        summary_dir = os.path.join(self.output_dir, "all_models_summary")
        os.makedirs(summary_dir, exist_ok=True)

        # Create summary dataframe
        summary_data = []
        for metrics in all_metrics:
            row = {
                'model_name': metrics.get('model_name', 'Unknown'),
                'model_file': metrics.get('model_file', 'Unknown'),
                'model_index': metrics.get('model_index', 0),
                'accuracy': f"{metrics.get('accuracy', 0):.4f}",
                'precision': f"{metrics.get('precision', 0):.4f}",
                'recall': f"{metrics.get('recall', 0):.4f}",
                'f1': f"{metrics.get('f1', 0):.4f}",
                'specificity': f"{metrics.get('specificity', 0):.4f}",
                'mcc': f"{metrics.get('mcc', 0):.4f}",
                'roc_auc': f"{metrics.get('roc_auc', 0):.4f}",
                'pr_auc': f"{metrics.get('pr_auc', 0):.4f}",
                'f1_from_filename': f"{metrics.get('f1_from_filename', 0):.4f}"
            }
            summary_data.append(row)

        # Save as CSV
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(summary_dir, "all_models_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

        # Save as Excel
        excel_path = os.path.join(summary_dir, "all_models_summary.xlsx")
        summary_df.to_excel(excel_path, index=False)

        # Save as text file (in English)
        txt_path = os.path.join(summary_dir, "all_models_summary.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("All Model Evaluation Results Summary\n")
            f.write("=" * 80 + "\n")
            f.write(f"Evaluation Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Models: {len(all_metrics)}\n")
            f.write("=" * 80 + "\n\n")

            # Sort by F1 score
            sorted_metrics = sorted(all_metrics, key=lambda x: x.get('f1', 0), reverse=True)

            f.write("Model Performance Ranking (by F1 Score):\n")
            f.write("-" * 80 + "\n")
            for i, metrics in enumerate(sorted_metrics, 1):
                f.write(f"{i:2d}. {metrics['model_name']:15} ({metrics['model_file'][:30]})\n")
                f.write(f"    F1: {metrics['f1']:.4f}, "
                        f"Accuracy: {metrics['accuracy']:.4f}, "
                        f"AUC: {metrics['roc_auc']:.4f}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Detailed Performance Metrics:\n")
            f.write("-" * 80 + "\n")
            for row in summary_data:
                f.write(f"{row['model_name']}:\n")
                f.write(f"  Accuracy: {row['accuracy']}, Precision: {row['precision']}, Recall: {row['recall']}\n")
                f.write(f"  F1 Score: {row['f1']}, Specificity: {row['specificity']}, MCC: {row['mcc']}\n")
                f.write(f"  ROC AUC: {row['roc_auc']}, PR AUC: {row['pr_auc']}\n")
                f.write(f"  Filename F1: {row['f1_from_filename']}\n")

        # Generate summary visualizations
        self._generate_summary_visualizations(summary_dir, all_metrics)

        print(f"📊 All model summary results saved to: {summary_dir}")

    def _generate_summary_visualizations(self, save_dir, all_metrics):
        """Generate summary visualization charts in English"""
        plt.style.use('seaborn-v0_8-whitegrid')

        # Prepare data
        model_names = [m['model_name'] for m in all_metrics]
        f1_scores = [m['f1'] for m in all_metrics]
        acc_scores = [m['accuracy'] for m in all_metrics]
        auc_scores = [m['roc_auc'] for m in all_metrics]
        f1_from_filenames = [m.get('f1_from_filename', 0) for m in all_metrics]

        # 1. Model Performance Comparison Bar Chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # F1 Score Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(model_names)), f1_scores, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('F1 Score', fontsize=12)
        ax1.set_title('F1 Score Comparison Across Models', fontsize=14)
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        # Display values on bars
        for bar, score in zip(bars1, f1_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{score:.4f}', ha='center', va='bottom', fontsize=9)

        # Accuracy Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(model_names)), acc_scores, color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Accuracy Comparison Across Models', fontsize=14)
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        for bar, score in zip(bars2, acc_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{score:.4f}', ha='center', va='bottom', fontsize=9)

        # AUC Comparison
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(model_names)), auc_scores, color='lightcoral', alpha=0.7)
        ax3.set_xlabel('Model', fontsize=12)
        ax3.set_ylabel('ROC AUC', fontsize=12)
        ax3.set_title('ROC AUC Comparison Across Models', fontsize=14)
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        for bar, score in zip(bars3, auc_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{score:.4f}', ha='center', va='bottom', fontsize=9)

        # Filename F1 vs Actual F1 Comparison
        ax4 = axes[1, 1]
        x = np.arange(len(model_names))
        width = 0.35
        bars4a = ax4.bar(x - width / 2, f1_scores, width, label='Actual F1', alpha=0.7)
        bars4b = ax4.bar(x + width / 2, f1_from_filenames, width, label='Filename F1', alpha=0.7)
        ax4.set_xlabel('Model', fontsize=12)
        ax4.set_ylabel('F1 Score', fontsize=12)
        ax4.set_title('Actual F1 vs Filename F1 Comparison', fontsize=14)
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        comparison_path = os.path.join(save_dir, "model_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Model Performance Scatter Matrix
        fig = plt.figure(figsize=(12, 10))

        metrics_matrix = np.array([f1_scores, acc_scores, auc_scores]).T
        metrics_names = ['F1 Score', 'Accuracy', 'ROC AUC']

        for i in range(3):
            for j in range(3):
                ax = plt.subplot(3, 3, i * 3 + j + 1)
                if i == j:
                    # Diagonal: histogram
                    ax.hist(metrics_matrix[:, i], bins=15, alpha=0.7,
                            color=['skyblue', 'lightgreen', 'lightcoral'][i])
                    ax.set_title(f'{metrics_names[i]} Distribution', fontsize=10)
                    ax.set_xlabel(metrics_names[i], fontsize=9)
                    ax.set_ylabel('Frequency', fontsize=9)
                else:
                    # Off-diagonal: scatter plot
                    ax.scatter(metrics_matrix[:, j], metrics_matrix[:, i],
                               alpha=0.6, s=50)
                    ax.set_xlabel(metrics_names[j], fontsize=9)
                    ax.set_ylabel(metrics_names[i], fontsize=9)

                    # Add regression line
                    if len(metrics_matrix) > 1:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            metrics_matrix[:, j], metrics_matrix[:, i]
                        )
                        x_range = np.array([min(metrics_matrix[:, j]), max(metrics_matrix[:, j])])
                        ax.plot(x_range, slope * x_range + intercept,
                                'r-', alpha=0.7, linewidth=1.5,
                                label=f'r={r_value:.3f}')
                        ax.legend(fontsize=8)

                ax.grid(True, alpha=0.3)

        plt.suptitle('Model Performance Metrics Relationship Matrix', fontsize=14)
        plt.tight_layout()
        matrix_path = os.path.join(save_dir, "performance_matrix.png")
        plt.savefig(matrix_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Model Ranking Heatmap
        plt.figure(figsize=(12, 8))

        # Prepare ranking data
        metrics_for_ranking = {
            'F1': f1_scores,
            'Accuracy': acc_scores,
            'AUC': auc_scores,
            'Precision': [m['precision'] for m in all_metrics],
            'Recall': [m['recall'] for m in all_metrics]
        }

        # Calculate rankings (1 is best)
        ranks = {}
        for metric_name, scores in metrics_for_ranking.items():
            # For metrics where higher is better
            ranks[metric_name] = stats.rankdata([-s for s in scores])  # Negative to rank highest as 1

        rank_df = pd.DataFrame(ranks, index=model_names)

        # Plot heatmap
        sns.heatmap(rank_df, annot=True, fmt='.0f', cmap='YlOrRd_r',
                    cbar_kws={'label': 'Rank (1 is Best)'})
        plt.title('Model Rankings Across Different Metrics', fontsize=14)
        plt.xlabel('Metric', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.tight_layout()
        rank_path = os.path.join(save_dir, "model_rankings.png")
        plt.savefig(rank_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📈 Summary visualization charts saved to: {save_dir}")

    def evaluate_ensemble(self, all_metrics, all_predictions):
        """Evaluate ensemble model"""
        print(f"\n{'=' * 80}")
        print("Starting Ensemble Model Evaluation")
        print(f"{'=' * 80}")

        # Create ensemble evaluation directory
        ensemble_dir = os.path.join(self.output_dir, "ensemble_evaluation")
        os.makedirs(ensemble_dir, exist_ok=True)

        # Merge all prediction results
        if not all_predictions:
            print("❌ No prediction results available for ensemble evaluation")
            return

        combined_predictions = pd.concat(all_predictions, ignore_index=True)

        # Group by sequence and calculate average probability
        ensemble_probs = combined_predictions.groupby(['sequence', 'true_label'])['probability'].mean().reset_index()
        ensemble_preds = (ensemble_probs['probability'] > 0.5).astype(int)

        # Calculate ensemble model metrics
        y_true = ensemble_probs['true_label'].values
        y_pred = ensemble_preds.values
        y_prob = ensemble_probs['probability'].values

        ensemble_metrics = self._calculate_detailed_metrics(y_true, y_pred, y_prob)

        # Save ensemble model results
        ensemble_results_df = pd.DataFrame({
            'sequence': ensemble_probs['sequence'],
            'true_label': y_true,
            'predicted_label': y_pred,
            'probability': y_prob,
            'correct': (y_true == y_pred).astype(int)
        })

        ensemble_results_path = os.path.join(ensemble_dir, "ensemble_predictions.csv")
        ensemble_results_df.to_csv(ensemble_results_path, index=False)

        # Save ensemble model metrics
        ensemble_metrics_path = os.path.join(ensemble_dir, "ensemble_metrics.txt")
        with open(ensemble_metrics_path, 'w') as f:
            f.write("Ensemble Model Evaluation Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Number of Models in Ensemble: {len(all_metrics)}\n")
            f.write(f"Evaluation Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy: {ensemble_metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {ensemble_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {ensemble_metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {ensemble_metrics['f1']:.4f}\n")
            f.write(f"  Specificity: {ensemble_metrics['specificity']:.4f}\n")
            f.write(f"  MCC: {ensemble_metrics['mcc']:.4f}\n")
            f.write(f"  ROC AUC: {ensemble_metrics['roc_auc']:.4f}\n")
            f.write(f"  PR AUC: {ensemble_metrics['pr_auc']:.4f}\n")

        # Generate ensemble model visualizations
        self._generate_visualizations(ensemble_dir, "Ensemble Model", y_true, y_pred, y_prob, ensemble_metrics)

        print(f"\n✅ Ensemble model evaluation completed!")
        print(f"   Number of models in ensemble: {len(all_metrics)}")
        print(f"   Ensemble model F1: {ensemble_metrics['f1']:.4f}")
        print(f"   Results saved in: {ensemble_dir}")


def main():
    """Main function"""
    print("=" * 80)
    print("Antimicrobial Peptide Prediction Model Batch Evaluation Tool")
    print("=" * 80)

    # Set paths
    checkpoints_dir = "./checkpoints"
    output_dir = "./evaluation_results"

    # Check if checkpoints directory exists
    if not os.path.exists(checkpoints_dir):
        print(f"❌ Error: Checkpoints directory not found: {checkpoints_dir}")
        return

    # Create evaluator
    evaluator = ModelEvaluator(checkpoints_dir, output_dir)

    # Start evaluation
    evaluator.evaluate_all_models()

    print("\n🎉 All evaluation tasks completed!")
    print(f"Please check the evaluation results in {output_dir} directory")


if __name__ == "__main__":
    main()
