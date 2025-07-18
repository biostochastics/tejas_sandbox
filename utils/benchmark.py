"""
Comprehensive Benchmark Suite - Tejas vs BERT vs Word2Vec
=========================================================

Generates publication-quality plots and metrics for research paper.
Tests memory usage, speed, accuracy, and pattern preservation.

"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd

# For comparison models
try:
    from gensim.models import Word2Vec
    from gensim.models.keyedvectors import KeyedVectors
    WORD2VEC_AVAILABLE = True
except ImportError:
    WORD2VEC_AVAILABLE = False
    print("Warning: gensim not available for Word2Vec comparison")

try:
    from transformers import AutoTokenizer, AutoModel
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: transformers not available for BERT comparison")

# Import our modules
from core.encoder import GoldenRatioEncoder
from core.fingerprint import BinaryFingerprintSearch
from core.decoder import SemanticDecoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set publication-quality plot parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (8, 6)


class BenchmarkSuite:
    """
    Comprehensive benchmark suite comparing Tejas, BERT, and Word2Vec.
    """
    
    def __init__(self, 
                 data_dir: str = "data/wikipedia",
                 model_dir: str = "models/fingerprint_encoder",
                 output_dir: str = "benchmark_results"):
        """
        Initialize benchmark suite.
        
        Args:
            data_dir: Directory containing Wikipedia data
            model_dir: Directory containing trained Tejas model
            output_dir: Directory for benchmark results
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different plot types
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.results = {}
        
        logger.info(f"Initialized BenchmarkSuite")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_test_data(self, n_samples: int = 10000) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Load test data with pattern families for evaluation.
        
        Returns:
            titles: List of all test titles
            pattern_families: Dict mapping patterns to title lists
        """
        logger.info(f"Loading test data (n_samples={n_samples})...")
        
        # Load titles
        titles_file = self.data_dir / "wikipedia_en_20231101_titles.pt"
        if titles_file.exists():
            data = torch.load(titles_file)
            all_titles = data['titles'] if isinstance(data, dict) else data
        else:
            raise FileNotFoundError(f"Wikipedia titles not found at {titles_file}")
        
        # Sample titles
        if n_samples < len(all_titles):
            indices = np.random.choice(len(all_titles), n_samples, replace=False)
            titles = [all_titles[i] for i in indices]
        else:
            titles = all_titles[:n_samples]
        
        # Organize by pattern families
        pattern_families = {
            'University': [],
            'List of': [],
            'History of': [],
            'Battle of': [],
            '(disambiguation)': [],
            '(film)': [],
            '(album)': [],
            'County': []
        }
        
        for title in titles:
            for pattern in pattern_families:
                if pattern in title:
                    pattern_families[pattern].append(title)
                    break
        
        logger.info(f"Loaded {len(titles)} titles")
        for pattern, members in pattern_families.items():
            logger.info(f"  {pattern}: {len(members)} titles")
        
        return titles, pattern_families
    
    def benchmark_tejas(self, titles: List[str], pattern_families: Dict[str, List[str]]) -> Dict:
        """Benchmark Tejas binary fingerprint system."""
        logger.info("\n" + "="*60)
        logger.info("BENCHMARKING TEJAS")
        logger.info("="*60)
        
        results = {}
        
        # Load pre-trained model
        encoder = GoldenRatioEncoder()
        encoder.load(self.model_dir)
        
        # Memory usage
        fingerprints_file = self.model_dir / "fingerprints.pt"
        if fingerprints_file.exists():
            data = torch.load(fingerprints_file)
            full_fingerprints = data['fingerprints']
            full_titles = data['titles']
            memory_mb = full_fingerprints.numel() * full_fingerprints.element_size() / 1024**2
        else:
            # Encode test titles
            fingerprints = encoder.encode(titles, batch_size=1000)
            memory_mb = fingerprints.numel() * fingerprints.element_size() / 1024**2
            full_fingerprints = fingerprints
            full_titles = titles
        
        results['memory_mb'] = memory_mb
        logger.info(f"Memory usage: {memory_mb:.2f} MB")
        
        # Encoding speed
        sample_titles = np.random.choice(titles, 100).tolist()
        start_time = time.time()
        _ = encoder.encode(sample_titles, show_progress=False)
        encode_time = time.time() - start_time
        results['encode_time_per_title'] = encode_time / len(sample_titles)
        logger.info(f"Encoding speed: {1/results['encode_time_per_title']:.0f} titles/sec")
        
        # Search speed
        search_engine = BinaryFingerprintSearch(full_fingerprints, full_titles)
        
        search_times = []
        for _ in range(100):
            query_idx = np.random.randint(len(titles))
            query = titles[query_idx]
            start_time = time.time()
            _ = search_engine.search(encoder.encode_single(query), k=10, show_pattern_analysis=False)
            search_times.append(time.time() - start_time)
        
        results['search_time_ms'] = np.mean(search_times) * 1000
        results['search_std_ms'] = np.std(search_times) * 1000
        logger.info(f"Search time: {results['search_time_ms']:.2f} ± {results['search_std_ms']:.2f} ms")
        
        # Pattern preservation accuracy
        pattern_accuracies = {}
        for pattern, pattern_titles in pattern_families.items():
            if len(pattern_titles) >= 2:
                # Test if pattern members are similar
                test_title = pattern_titles[0]
                query_fp = encoder.encode_single(test_title)
                search_results = search_engine.search(query_fp, k=20, show_pattern_analysis=False)
                
                # Count how many results share the pattern
                pattern_count = sum(1 for title, _, _ in search_results if pattern in title)
                accuracy = pattern_count / len(search_results)
                pattern_accuracies[pattern] = accuracy
        
        results['pattern_accuracies'] = pattern_accuracies
        results['avg_pattern_accuracy'] = np.mean(list(pattern_accuracies.values()))
        logger.info(f"Average pattern accuracy: {results['avg_pattern_accuracy']:.3f}")
        
        # False positive rate (searching for pattern that shouldn't match)
        nonsense_query = "xyzqwerty123nonsense"
        query_fp = encoder.encode_single(nonsense_query)
        search_results = search_engine.search(query_fp, k=100, show_pattern_analysis=False)
        
        # Check if any results actually contain the nonsense string
        false_positives = sum(1 for title, _, _ in search_results if nonsense_query.lower() in title.lower())
        results['false_positive_rate'] = false_positives / len(search_results)
        logger.info(f"False positive rate: {results['false_positive_rate']:.3%}")
        
        return results
    
    def benchmark_word2vec(self, titles: List[str], pattern_families: Dict[str, List[str]]) -> Dict:
        """Benchmark Word2Vec."""
        if not WORD2VEC_AVAILABLE:
            logger.warning("Word2Vec not available, skipping benchmark")
            return {}
        
        logger.info("\n" + "="*60)
        logger.info("BENCHMARKING WORD2VEC")
        logger.info("="*60)
        
        results = {}
        
        # Prepare data for Word2Vec (tokenize titles)
        tokenized_titles = [title.lower().split() for title in titles]
        
        # Train Word2Vec model
        logger.info("Training Word2Vec model...")
        start_time = time.time()
        model = Word2Vec(
            sentences=tokenized_titles,
            vector_size=300,
            window=5,
            min_count=1,
            workers=4,
            epochs=5
        )
        train_time = time.time() - start_time
        results['train_time'] = train_time
        logger.info(f"Training time: {train_time:.2f}s")
        
        # Memory usage (approximate)
        n_words = len(model.wv)
        memory_mb = n_words * 300 * 4 / 1024**2  # 300 dims, float32
        results['memory_mb'] = memory_mb
        logger.info(f"Memory usage: {memory_mb:.2f} MB")
        
        # Create title embeddings (average word vectors)
        title_embeddings = []
        for tokens in tokenized_titles:
            valid_tokens = [t for t in tokens if t in model.wv]
            if valid_tokens:
                embedding = np.mean([model.wv[t] for t in valid_tokens], axis=0)
            else:
                embedding = np.zeros(300)
            title_embeddings.append(embedding)
        title_embeddings = np.array(title_embeddings)
        
        # Search speed
        search_times = []
        for _ in range(100):
            query_idx = np.random.randint(len(titles))
            query_embedding = title_embeddings[query_idx]
            
            start_time = time.time()
            similarities = cosine_similarity([query_embedding], title_embeddings)[0]
            top_k = np.argsort(similarities)[-10:][::-1]
            search_times.append(time.time() - start_time)
        
        results['search_time_ms'] = np.mean(search_times) * 1000
        results['search_std_ms'] = np.std(search_times) * 1000
        logger.info(f"Search time: {results['search_time_ms']:.2f} ± {results['search_std_ms']:.2f} ms")
        
        # Pattern preservation accuracy
        pattern_accuracies = {}
        for pattern, pattern_titles in pattern_families.items():
            if len(pattern_titles) >= 2:
                # Get embedding for first pattern title
                pattern_idx = titles.index(pattern_titles[0])
                query_embedding = title_embeddings[pattern_idx]
                
                # Find similar titles
                similarities = cosine_similarity([query_embedding], title_embeddings)[0]
                top_20_idx = np.argsort(similarities)[-20:][::-1]
                top_20_titles = [titles[i] for i in top_20_idx]
                
                # Count pattern matches
                pattern_count = sum(1 for t in top_20_titles if pattern in t)
                accuracy = pattern_count / len(top_20_titles)
                pattern_accuracies[pattern] = accuracy
        
        results['pattern_accuracies'] = pattern_accuracies
        results['avg_pattern_accuracy'] = np.mean(list(pattern_accuracies.values()))
        logger.info(f"Average pattern accuracy: {results['avg_pattern_accuracy']:.3f}")
        
        return results
    
    def benchmark_bert(self, titles: List[str], pattern_families: Dict[str, List[str]], 
                      sample_size: int = 1000) -> Dict:
        """Benchmark BERT (on smaller sample due to computational cost)."""
        if not BERT_AVAILABLE:
            logger.warning("BERT not available, skipping benchmark")
            return {}
        
        logger.info("\n" + "="*60)
        logger.info("BENCHMARKING BERT")
        logger.info("="*60)
        
        results = {}
        
        # Use smaller sample for BERT
        if len(titles) > sample_size:
            sample_idx = np.random.choice(len(titles), sample_size, replace=False)
            sample_titles = [titles[i] for i in sample_idx]
        else:
            sample_titles = titles
        
        # Load BERT model
        logger.info("Loading BERT model...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')
        model.eval()
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Memory usage (model + embeddings)
        model_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        embedding_memory_mb = len(titles) * 768 * 4 / 1024**2  # 768 dims, float32
        results['memory_mb'] = model_memory_mb + embedding_memory_mb
        logger.info(f"Memory usage: {results['memory_mb']:.2f} MB")
        
        # Encoding speed
        encode_times = []
        batch_size = 32
        
        for i in range(0, min(100, len(sample_titles)), batch_size):
            batch = sample_titles[i:i+batch_size]
            
            start_time = time.time()
            with torch.no_grad():
                inputs = tokenizer(batch, padding=True, truncation=True, 
                                 max_length=128, return_tensors='pt').to(device)
                outputs = model(**inputs)
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            encode_times.append((time.time() - start_time) / len(batch))
        
        results['encode_time_per_title'] = np.mean(encode_times)
        logger.info(f"Encoding speed: {1/results['encode_time_per_title']:.1f} titles/sec")
        
        # Create embeddings for search test
        logger.info("Creating embeddings for search test...")
        title_embeddings = []
        
        for i in tqdm(range(0, len(sample_titles), batch_size)):
            batch = sample_titles[i:i+batch_size]
            with torch.no_grad():
                inputs = tokenizer(batch, padding=True, truncation=True,
                                 max_length=128, return_tensors='pt').to(device)
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                title_embeddings.extend(batch_embeddings)
        
        title_embeddings = np.array(title_embeddings)
        
        # Search speed
        search_times = []
        for _ in range(50):  # Fewer searches due to cost
            query_idx = np.random.randint(len(sample_titles))
            query_embedding = title_embeddings[query_idx]
            
            start_time = time.time()
            similarities = cosine_similarity([query_embedding], title_embeddings)[0]
            top_k = np.argsort(similarities)[-10:][::-1]
            search_times.append(time.time() - start_time)
        
        results['search_time_ms'] = np.mean(search_times) * 1000
        results['search_std_ms'] = np.std(search_times) * 1000
        logger.info(f"Search time: {results['search_time_ms']:.2f} ± {results['search_std_ms']:.2f} ms")
        
        # Pattern preservation (on subset)
        pattern_accuracies = {}
        for pattern, pattern_titles in pattern_families.items():
            pattern_titles_in_sample = [t for t in pattern_titles if t in sample_titles]
            if len(pattern_titles_in_sample) >= 2:
                # Get embedding for first pattern title
                pattern_idx = sample_titles.index(pattern_titles_in_sample[0])
                query_embedding = title_embeddings[pattern_idx]
                
                # Find similar titles
                similarities = cosine_similarity([query_embedding], title_embeddings)[0]
                top_20_idx = np.argsort(similarities)[-20:][::-1]
                top_20_titles = [sample_titles[i] for i in top_20_idx]
                
                # Count pattern matches
                pattern_count = sum(1 for t in top_20_titles if pattern in t)
                accuracy = pattern_count / len(top_20_titles)
                pattern_accuracies[pattern] = accuracy
        
        results['pattern_accuracies'] = pattern_accuracies
        results['avg_pattern_accuracy'] = np.mean(list(pattern_accuracies.values()))
        logger.info(f"Average pattern accuracy: {results['avg_pattern_accuracy']:.3f}")
        
        return results
    
    def generate_confusion_matrix(self, titles: List[str], pattern_families: Dict[str, List[str]]):
        """Generate confusion matrix for Tejas pattern classification."""
        logger.info("\nGenerating confusion matrix for Tejas...")
        
        # Load Tejas model
        encoder = GoldenRatioEncoder()
        encoder.load(self.model_dir)
        
        # Load fingerprint database
        data = torch.load(self.model_dir / "fingerprints.pt")
        search_engine = BinaryFingerprintSearch(data['fingerprints'], data['titles'])
        
        # Prepare test data
        test_patterns = list(pattern_families.keys())
        y_true = []
        y_pred = []
        
        # Sample titles from each pattern
        samples_per_pattern = 50
        for true_pattern in test_patterns:
            pattern_titles = pattern_families[true_pattern][:samples_per_pattern]
            
            for title in pattern_titles:
                if title in data['titles']:  # Only test if in database
                    # Get search results
                    query_fp = encoder.encode_single(title)
                    results = search_engine.search(query_fp, k=5, show_pattern_analysis=False)
                    
                    # Determine predicted pattern based on top results
                    pattern_counts = {p: 0 for p in test_patterns}
                    for result_title, _, _ in results[1:]:  # Skip self
                        for pattern in test_patterns:
                            if pattern in result_title:
                                pattern_counts[pattern] += 1
                                break
                    
                    # Predict pattern with highest count
                    pred_pattern = max(pattern_counts, key=pattern_counts.get)
                    if pattern_counts[pred_pattern] == 0:
                        pred_pattern = "Other"
                    
                    y_true.append(true_pattern)
                    y_pred.append(pred_pattern)
        
        # Add "Other" to patterns if needed
        if "Other" in y_pred:
            test_patterns.append("Other")
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=test_patterns)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=test_patterns, yticklabels=test_patterns)
        plt.title('Tejas Pattern Classification Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Pattern', fontsize=14)
        plt.ylabel('True Pattern', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix_tejas.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate metrics
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        logger.info(f"Pattern classification accuracy: {accuracy:.3f}")
        
        # Save classification report
        report = classification_report(y_true, y_pred, labels=test_patterns, output_dict=True)
        with open(self.output_dir / 'classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return cm, accuracy
    
    def plot_memory_comparison(self, results: Dict):
        """Generate memory usage comparison plot."""
        systems = ['Tejas', 'Word2Vec', 'BERT']
        memories = []
        
        for system in systems:
            if system in results and 'memory_mb' in results[system]:
                memories.append(results[system]['memory_mb'])
            else:
                memories.append(0)
        
        # Create bar plot
        plt.figure(figsize=(8, 6))
        bars = plt.bar(systems, memories, color=['#2E86AB', '#A23B72', '#F18F01'])
        
        # Add value labels
        for bar, mem in zip(bars, memories):
            if mem > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        f'{mem:.0f} MB', ha='center', va='bottom', fontsize=12)
        
        plt.ylabel('Memory Usage (MB)', fontsize=14)
        plt.title('Memory Usage Comparison', fontsize=16)
        plt.ylim(0, max(memories) * 1.2)
        
        # Add grid
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'memory_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved memory comparison plot")
    
    def plot_search_speed_comparison(self, results: Dict):
        """Generate search speed comparison plot."""
        systems = []
        search_times = []
        search_stds = []
        
        for system in ['Tejas', 'Word2Vec', 'BERT']:
            if system in results and 'search_time_ms' in results[system]:
                systems.append(system)
                search_times.append(results[system]['search_time_ms'])
                search_stds.append(results[system].get('search_std_ms', 0))
        
        # Create bar plot with error bars
        plt.figure(figsize=(8, 6))
        x = np.arange(len(systems))
        bars = plt.bar(x, search_times, yerr=search_stds, 
                       color=['#2E86AB', '#A23B72', '#F18F01'][:len(systems)],
                       capsize=5)
        
        # Add value labels
        for i, (bar, time) in enumerate(zip(bars, search_times)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + search_stds[i] + 0.5,
                    f'{time:.1f} ms', ha='center', va='bottom', fontsize=12)
        
        plt.ylabel('Search Time (ms)', fontsize=14)
        plt.title('Search Speed Comparison', fontsize=16)
        plt.xticks(x, systems)
        plt.yscale('log')  # Log scale for better visibility
        
        # Add grid
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'search_speed_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved search speed comparison plot")
    
    def plot_pattern_accuracy_comparison(self, results: Dict):
        """Generate pattern preservation accuracy comparison."""
        systems = []
        accuracies = []
        
        for system in ['Tejas', 'Word2Vec', 'BERT']:
            if system in results and 'avg_pattern_accuracy' in results[system]:
                systems.append(system)
                accuracies.append(results[system]['avg_pattern_accuracy'])
        
        # Create bar plot
        plt.figure(figsize=(8, 6))
        bars = plt.bar(systems, accuracies, 
                       color=['#2E86AB', '#A23B72', '#F18F01'][:len(systems)])
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=12)
        
        plt.ylabel('Pattern Preservation Accuracy', fontsize=14)
        plt.title('Pattern Preservation Comparison', fontsize=16)
        plt.ylim(0, 1.1)
        
        # Add grid
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'pattern_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved pattern accuracy comparison plot")
    
    def plot_detailed_pattern_accuracy(self, results: Dict):
        """Generate detailed pattern accuracy plot for each system."""
        for system in ['Tejas', 'Word2Vec', 'BERT']:
            if system not in results or 'pattern_accuracies' not in results[system]:
                continue
            
            pattern_acc = results[system]['pattern_accuracies']
            if not pattern_acc:
                continue
            
            patterns = list(pattern_acc.keys())
            accuracies = list(pattern_acc.values())
            
            # Create horizontal bar plot
            plt.figure(figsize=(10, 6))
            y_pos = np.arange(len(patterns))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(patterns)))
            bars = plt.barh(y_pos, accuracies, color=colors)
            
            # Add value labels
            for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{acc:.3f}', va='center', fontsize=10)
            
            plt.yticks(y_pos, patterns)
            plt.xlabel('Accuracy', fontsize=14)
            plt.title(f'{system} - Pattern-wise Accuracy', fontsize=16)
            plt.xlim(0, 1.15)
            
            # Add grid
            plt.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'pattern_accuracy_{system.lower()}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved detailed pattern accuracy plot for {system}")
    
    def plot_speedup_factors(self, results: Dict):
        """Generate speedup factor comparison plot."""
        if 'Tejas' not in results:
            return
        
        tejas_search = results['Tejas']['search_time_ms']
        tejas_memory = results['Tejas']['memory_mb']
        
        metrics = ['Search Speed', 'Memory Efficiency']
        word2vec_factors = []
        bert_factors = []
        
        # Calculate speedup factors
        if 'Word2Vec' in results:
            word2vec_factors.append(results['Word2Vec']['search_time_ms'] / tejas_search)
            word2vec_factors.append(results['Word2Vec']['memory_mb'] / tejas_memory)
        
        if 'BERT' in results:
            bert_factors.append(results['BERT']['search_time_ms'] / tejas_search)
            bert_factors.append(results['BERT']['memory_mb'] / tejas_memory)
        
        # Create grouped bar plot
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        
        if word2vec_factors:
            bars1 = plt.bar(x - width/2, word2vec_factors, width, 
                           label='vs Word2Vec', color='#A23B72')
            # Add value labels
            for bar, val in zip(bars1, word2vec_factors):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}x', ha='center', va='bottom', fontsize=12)
        
        if bert_factors:
            bars2 = plt.bar(x + width/2, bert_factors, width, 
                           label='vs BERT', color='#F18F01')
            # Add value labels
            for bar, val in zip(bars2, bert_factors):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}x', ha='center', va='bottom', fontsize=12)
        
        plt.ylabel('Speedup Factor', fontsize=14)
        plt.title('Tejas Performance Advantage', fontsize=16)
        plt.xticks(x, metrics)
        plt.legend()
        plt.yscale('log')
        
        # Add horizontal line at y=1
        plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        # Add grid
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'speedup_factors.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved speedup factors plot")
    
    def generate_summary_table(self, results: Dict):
        """Generate a summary table of all metrics."""
        metrics = ['Memory (MB)', 'Search Time (ms)', 'Pattern Accuracy', 'False Positive Rate']
        systems = ['Tejas', 'Word2Vec', 'BERT']
        
        data = []
        for system in systems:
            if system not in results:
                data.append(['-'] * len(metrics))
                continue
            
            row = []
            res = results[system]
            
            # Memory
            row.append(f"{res.get('memory_mb', 0):.1f}" if 'memory_mb' in res else '-')
            
            # Search time
            if 'search_time_ms' in res:
                row.append(f"{res['search_time_ms']:.2f} ± {res.get('search_std_ms', 0):.2f}")
            else:
                row.append('-')
            
            # Pattern accuracy
            row.append(f"{res.get('avg_pattern_accuracy', 0):.3f}" if 'avg_pattern_accuracy' in res else '-')
            
            # False positive rate (only for Tejas)
            row.append(f"{res.get('false_positive_rate', 0):.3%}" if system == 'Tejas' else 'N/A')
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=metrics, index=systems)
        
        # Save as CSV
        df.to_csv(self.output_dir / 'benchmark_summary.csv')
        
        # Create visual table
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        rowLabels=df.index,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(systems)):
            table[(i+1, -1)].set_facecolor('#E8E8E8')
        for j in range(len(metrics)):
            table[(0, j)].set_facecolor('#D0D0D0')
        
        plt.title('Benchmark Summary', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'benchmark_summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved benchmark summary table")
        
        return df
    
    def run_complete_benchmark(self, n_samples: int = 10000):
        """Run complete benchmark suite."""
        logger.info("="*80)
        logger.info("STARTING COMPLETE BENCHMARK SUITE")
        logger.info("="*80)
        
        # Load test data
        titles, pattern_families = self.load_test_data(n_samples)
        
        # Run benchmarks
        results = {}
        
        # Tejas benchmark
        results['Tejas'] = self.benchmark_tejas(titles, pattern_families)
        
        # Word2Vec benchmark
        if WORD2VEC_AVAILABLE:
            results['Word2Vec'] = self.benchmark_word2vec(titles, pattern_families)
        
        # BERT benchmark (on smaller sample)
        if BERT_AVAILABLE:
            results['BERT'] = self.benchmark_bert(titles, pattern_families, sample_size=1000)
        
        # Save raw results
        with open(self.output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate plots
        logger.info("\nGenerating plots...")
        self.plot_memory_comparison(results)
        self.plot_search_speed_comparison(results)
        self.plot_pattern_accuracy_comparison(results)
        self.plot_detailed_pattern_accuracy(results)
        self.plot_speedup_factors(results)
        
        # Generate confusion matrix for Tejas
        cm, accuracy = self.generate_confusion_matrix(titles, pattern_families)
        
        # Generate summary table
        summary_df = self.generate_summary_table(results)
        
        logger.info("\n" + "="*80)
        logger.info("BENCHMARK COMPLETE")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*80)
        
        return results, summary_df


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Tejas vs BERT vs Word2Vec")
    parser.add_argument("--data-dir", default="data/wikipedia",
                       help="Directory containing Wikipedia data")
    parser.add_argument("--model-dir", default="models/fingerprint_encoder",
                       help="Directory containing trained Tejas model")
    parser.add_argument("--output-dir", default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--n-samples", type=int, default=10000,
                       help="Number of titles to use for testing")
    
    args = parser.parse_args()
    
    # Create benchmark suite
    benchmark = BenchmarkSuite(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    # Run benchmarks
    results, summary = benchmark.run_complete_benchmark(n_samples=args.n_samples)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(summary)


if __name__ == "__main__":
    main()