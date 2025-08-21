#!/usr/bin/env python3
"""
Tejas: Quantum Semantic Fingerprint Framework
============================================

Unified entry point for training and searching with consciousness-aligned fingerprints.

Usage:
    Training:
        python run.py --mode train --dataset path/to/data.pt --output models/my_model
    
    Demo (Interactive Search):
        python run.py --mode demo --model models/my_model

Author: Quantum Semantic Framework
"""

import argparse
import sys
import logging
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Tejas: Quantum Semantic Fingerprint Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train on Wikipedia dataset with bit packing:
    python run.py --mode train --dataset data/wikipedia/wikipedia_en_20231101_titles.pt --bits 128 --pack-bits --backend auto
    
  Run interactive search demo:
    python run.py --mode demo --model models/fingerprint_encoder --backend numba
    
  Run demo with specific query:
    python run.py --mode demo --model models/fingerprint_encoder --query "quantum mechanics" --top-k 15
    
  Benchmark search performance:
    python run.py --mode benchmark --model models/fingerprint_encoder --benchmark-backend all --benchmark-sizes 1000,50000,100000
    
  Migrate V1 format to V2:
    python run.py --mode migrate --input models/old_model --output models/new_model.tj2 --target-version 2
    
  Run calibration analysis:
    python run.py --mode calibrate --model models/fingerprint_encoder --k-values 1,5,10 --n-folds 5
    
  Monitor for drift:
    python run.py --mode drift --model models/fingerprint_encoder --baseline data/baseline.pt --batch data/new_batch.pt
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True,
        choices=['train', 'demo', 'benchmark', 'migrate', 'calibrate', 'drift'],
        help='Operation mode: train, demo, benchmark, migrate, calibrate, or drift'
    )
    
    # Global arguments (used by multiple modes)
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['cpu', 'cuda', 'auto'],
        help='Device for computation (default: auto)'
    )
    
    # Training arguments
    train_group = parser.add_argument_group('Training options')
    train_group.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset file (required for training)'
    )
    train_group.add_argument(
        '--output',
        type=str,
        default='models/fingerprint_encoder',
        help='Output directory for trained model (default: models/fingerprint_encoder)'
    )
    train_group.add_argument(
        '--bits',
        type=int,
        default=128,
        help='Number of bits in fingerprint (default: 128)'
    )
    train_group.add_argument(
        '--max-features',
        type=int,
        default=10000,
        help='Maximum number of n-gram features (default: 10000)'
    )
    train_group.add_argument(
        '--memory-limit',
        type=int,
        default=50,
        help='Memory limit in GB for training (default: 50)'
    )
    train_group.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Batch size for encoding (default: 10000)'
    )
    train_group.add_argument(
        '--threshold-strategy',
        type=str,
        choices=['zero', 'median'],
        default='zero',
        help="Threshold strategy for binarization: 'zero' or 'median' (default: zero)"
    )
    train_group.add_argument(
        '--pack-bits',
        action='store_true',
        help='Enable bit packing for fingerprints (saved as packed bytes with metadata)'
    )
    train_group.add_argument(
        '--bitorder',
        type=str,
        default='little',
        choices=['little', 'big'],
        help="Bit order to use when packing ('little' or 'big'), default: little"
    )
    train_group.add_argument(
        '--backend',
        type=str,
        default='auto',
        choices=['numpy', 'numba', 'native', 'auto'],
        help='Backend for packed operations: numpy, numba, native, auto (default: auto)'
    )
    train_group.add_argument(
        '--format-version',
        type=int,
        default=2,
        choices=[1, 2],
        help='Output format version: 1 (legacy), 2 (binary with header) (default: 2)'
    )
    train_group.add_argument(
        '--max-titles',
        type=int,
        default=None,
        help='Maximum titles to use (for testing, default: use all)'
    )
    
    # Benchmark arguments
    bench_group = parser.add_argument_group('Benchmark options')
    bench_group.add_argument(
        '--benchmark-backend',
        type=str,
        default='all',
        choices=['numpy', 'numba', 'native', 'auto', 'all'],
        help='Backend(s) to benchmark: specific backend or "all" (default: all)'
    )
    bench_group.add_argument(
        '--benchmark-sizes',
        type=str,
        default='1000,10000,50000',
        help='Comma-separated database sizes to test (default: 1000,10000,50000)'
    )
    
    # Demo arguments
    demo_group = parser.add_argument_group('Demo options')
    demo_group.add_argument(
        '--model',
        type=str,
        default='models/fingerprint_encoder',
        help='Path to trained model directory (default: models/fingerprint_encoder)'
    )
    demo_group.add_argument(
        '--query',
        type=str,
        help='Search query (for non-interactive demo)'
    )
    demo_group.add_argument(
        '--pattern',
        type=str,
        help='Pattern to search for (e.g., "List of")'
    )
    demo_group.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )
    demo_group.add_argument(
        '--backend',
        type=str,
        default='auto',
        choices=['numpy', 'numba', 'native', 'auto'],
        help='Backend for search operations (default: auto)'
    )
    
    # Migration arguments
    migrate_group = parser.add_argument_group('Migration options')
    migrate_group.add_argument(
        '--input',
        type=str,
        help='Input path (V1 directory or V2 file)'
    )
    migrate_group.add_argument(
        '--output',
        type=str,
        help='Output path for migrated format'
    )
    migrate_group.add_argument(
        '--target-version',
        type=int,
        default=2,
        choices=[1, 2],
        help='Target format version (default: 2)'
    )
    
    # Calibration arguments
    calib_group = parser.add_argument_group('Calibration options')
    calib_group.add_argument(
        '--model',
        type=str,
        default='models/fingerprint_encoder',
        help='Path to trained model directory'
    )
    calib_group.add_argument(
        '--k-values',
        type=str,
        default='1,5,10,20',
        help='Comma-separated k values for Precision@k, Recall@k (default: 1,5,10,20)'
    )
    calib_group.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    calib_group.add_argument(
        '--n-bootstrap',
        type=int,
        default=100,
        help='Number of bootstrap samples for confidence intervals (default: 100)'
    )
    calib_group.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size for train/test split (default: 0.2)'
    )
    calib_group.add_argument(
        '--output-calibration',
        type=str,
        default='calibration_results.json',
        help='Output file for calibration results (default: calibration_results.json)'
    )
    
    # Drift monitoring arguments
    drift_group = parser.add_argument_group('Drift monitoring options')
    drift_group.add_argument(
        '--baseline',
        type=str,
        help='Path to baseline fingerprint data (required for drift mode)'
    )
    drift_group.add_argument(
        '--batch',
        type=str,
        help='Path to new batch data to check for drift'
    )
    drift_group.add_argument(
        '--js-threshold',
        type=float,
        default=0.1,
        help='Jensen-Shannon divergence threshold for drift detection (default: 0.1)'
    )
    drift_group.add_argument(
        '--ks-threshold',
        type=float,
        default=0.05,
        help='Kolmogorov-Smirnov test p-value threshold (default: 0.05)'
    )
    drift_group.add_argument(
        '--entropy-threshold',
        type=float,
        default=0.1,
        help='Entropy change threshold for drift detection (default: 0.1)'
    )
    drift_group.add_argument(
        '--output-drift',
        type=str,
        default='drift_report.json',
        help='Output file for drift analysis results (default: drift_report.json)'
    )
    
    try:
        args = parser.parse_args()
        
        # Validate arguments based on mode
        if args.mode == 'train':
            if not args.dataset:
                parser.error("--dataset is required for training mode")
            
            # Import and run training
            from train.wikipedia_train import WikipediaTrainer
            
            # Handle 'auto' device selection for training
            device = args.device
            if device == 'auto':
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Auto-selected device: {device}")
            
            trainer = WikipediaTrainer(
                n_bits=args.bits,
                max_features=args.max_features,
                output_dir=args.output,
                device=device,
                threshold_strategy=args.threshold_strategy,
                pack_bits=bool(args.pack_bits),
                bitorder=args.bitorder,
                backend=args.backend,
                format_version=args.format_version
            )
            
            logger.info(f"Starting training with dataset: {args.dataset}")
            trainer.train(
                dataset_path=args.dataset,
                memory_limit_gb=args.memory_limit,
                batch_size=args.batch_size,
                max_titles=args.max_titles
            )
            
        elif args.mode == 'demo':
            # Import and run demo
            from demo.wikipedia_demo import WikipediaDemo
            
            demo = WikipediaDemo(
                model_dir=args.model,
                device=args.device
            )
            
            if args.query:
                # Single query mode
                results = demo.search(args.query, k=args.top_k)
                demo.display_results(args.query, results)
            elif args.pattern:
                # Pattern search mode
                results = demo.search_pattern(args.pattern)
                demo.display_pattern_results(args.pattern, results)
            else:
                # Interactive mode
                demo.interactive()
                
        elif args.mode == 'benchmark':
            # Import and run benchmark
            from demo.wikipedia_demo import WikipediaDemo
            
            demo = WikipediaDemo(
                model_dir=args.model,
                device=args.device,
                backend=getattr(args, 'benchmark_backend', 'auto')
            )
            
            # Parse benchmark sizes
            try:
                sizes = [int(x.strip()) for x in args.benchmark_sizes.split(',')]
            except:
                sizes = [1000, 10000, 50000]
            
            demo.benchmark(n_queries=100, test_sizes=sizes, benchmark_backend=args.benchmark_backend)
            
        elif args.mode == 'migrate':
            # Import and run format migration
            from core.format import migrate_from_v1, detect_format_version
            
            if not args.input or not args.output:
                parser.error("--input and --output are required for migration mode")
            
            input_path = Path(args.input)
            output_path = Path(args.output)
            
            logger.info(f"Migrating {input_path} → {output_path}")
            
            # Detect input format
            try:
                input_version = detect_format_version(input_path)
                logger.info(f"Detected input format: V{input_version}")
            except Exception as e:
                logger.error(f"Could not detect input format: {e}")
                sys.exit(1)
            
            # Perform migration
            if input_version == 1 and args.target_version == 2:
                migrate_from_v1(input_path, output_path)
                logger.info("Migration completed successfully")
            else:
                logger.error(f"Migration from V{input_version} to V{args.target_version} not implemented")
                sys.exit(1)
                
        elif args.mode == 'calibrate':
            # Import and run calibration analysis
            from core.calibration import StatisticalCalibrator
            from core.encoder import GoldenRatioEncoder
            from core.fingerprint import BinaryFingerprintSearch
            import torch
            import json
            
            logger.info("Starting calibration analysis...")
            
            # Load model and fingerprints
            encoder = GoldenRatioEncoder()
            encoder.load(args.model)
            
            # Load fingerprint database
            fingerprints_file = Path(args.model) / "fingerprints.pt"
            if not fingerprints_file.exists():
                logger.error(f"Fingerprints not found at {fingerprints_file}")
                sys.exit(1)
            
            data = torch.load(fingerprints_file)
            titles = data['titles']
            
            # Handle packed/unpacked fingerprints
            if 'fingerprints' in data:
                fingerprints = data['fingerprints']
            elif 'fingerprints_packed' in data:
                from core.fingerprint import unpack_fingerprints
                n_bits = data.get('n_bits', encoder.n_bits)
                bitorder = data.get('bitorder', 'little')
                fingerprints = unpack_fingerprints(data['fingerprints_packed'], n_bits, bitorder)
            else:
                logger.error("No fingerprints found in data file")
                sys.exit(1)
            
            # Create pattern families for relevance assessment
            pattern_families = {
                'University': [t for t in titles if 'University' in t],
                'List of': [t for t in titles if 'List of' in t],
                'History of': [t for t in titles if 'History of' in t],
                'Battle of': [t for t in titles if 'Battle of' in t],
                '(disambiguation)': [t for t in titles if '(disambiguation)' in t],
                '(film)': [t for t in titles if '(film)' in t],
            }
            
            # Generate calibration data
            search_engine = BinaryFingerprintSearch(fingerprints, titles)
            calibrator = StatisticalCalibrator()
            
            similarity_scores = []
            relevance_scores = []
            
            # Sample queries from each pattern family
            import numpy as np
            np.random.seed(42)
            
            for pattern, pattern_titles in pattern_families.items():
                if len(pattern_titles) < 5:
                    continue
                
                # Sample queries from this pattern
                n_queries = min(20, len(pattern_titles) // 2)
                sample_queries = np.random.choice(pattern_titles, n_queries, replace=False)
                
                for query_title in sample_queries:
                    query_fp = encoder.encode_single(query_title)
                    results = search_engine.search(query_fp, k=50, show_pattern_analysis=False, backend=args.backend)
                    
                    query_similarities = []
                    query_relevance = []
                    
                    for result_title, distance, _ in results:
                        if result_title == query_title:
                            continue  # Skip self-match
                        
                        # Convert Hamming distance to similarity
                        similarity = 1.0 / (1.0 + float(distance))
                        query_similarities.append(similarity)
                        
                        # Relevance based on pattern match
                        relevance = 1 if pattern in result_title else 0
                        query_relevance.append(relevance)
                    
                    if len(query_similarities) > 0:
                        similarity_scores.append(query_similarities)
                        relevance_scores.append(query_relevance)
            
            if len(similarity_scores) == 0:
                logger.error("No valid queries found for calibration")
                sys.exit(1)
            
            # Parse k values
            k_values = [int(k.strip()) for k in args.k_values.split(',')]
            
            # Run calibration
            logger.info(f"Running calibration with {len(similarity_scores)} queries")
            result = calibrator.calibrate_with_cv(
                similarity_scores, relevance_scores,
                k_values=k_values,
                n_folds=args.n_folds,
                n_bootstrap=args.n_bootstrap,
                test_size=args.test_size
            )
            
            # Save results
            calibrator.save_results(result, args.output_calibration, format='json')
            
            # Display key metrics
            metrics = result['metrics']
            logger.info("\n" + "="*50)
            logger.info("CALIBRATION RESULTS")
            logger.info("="*50)
            logger.info(f"MAP: {metrics['map']:.3f}")
            logger.info(f"NDCG: {metrics['ndcg']:.3f}")
            logger.info(f"ROC AUC: {metrics.get('roc_auc', 0):.3f}")
            
            for k in k_values:
                if f'precision@{k}' in metrics:
                    logger.info(f"Precision@{k}: {metrics[f'precision@{k}']:.3f}")
                if f'recall@{k}' in metrics:
                    logger.info(f"Recall@{k}: {metrics[f'recall@{k}']:.3f}")
            
            logger.info(f"\nResults saved to: {args.output_calibration}")
            
        elif args.mode == 'drift':
            # Import and run drift monitoring
            from core.drift import DriftMonitor
            import torch
            import json
            
            if not args.baseline:
                parser.error("--baseline is required for drift mode")
            
            logger.info("Starting drift monitoring...")
            
            # Load baseline data
            baseline_data = torch.load(args.baseline)
            if isinstance(baseline_data, dict):
                # Extract fingerprints from saved format
                if 'fingerprints' in baseline_data:
                    baseline_fingerprints = baseline_data['fingerprints']
                elif 'fingerprints_packed' in baseline_data:
                    from core.fingerprint import unpack_fingerprints
                    n_bits = baseline_data.get('n_bits', 1024)
                    bitorder = baseline_data.get('bitorder', 'little')
                    baseline_fingerprints = unpack_fingerprints(
                        baseline_data['fingerprints_packed'], n_bits, bitorder
                    )
                else:
                    logger.error("No fingerprints found in baseline data")
                    sys.exit(1)
            else:
                baseline_fingerprints = baseline_data
            
            # Initialize drift monitor with supported parameters
            monitor = DriftMonitor(
                baseline_fingerprints=None,
                drift_threshold=args.ks_threshold,  # Use KS threshold as drift threshold
                sensitivity='medium',  # Can be made configurable if needed
                min_batch_size=50
            )
            monitor.set_baseline(baseline_fingerprints)
            
            results = []
            
            if args.batch:
                # Single batch analysis
                batch_data = torch.load(args.batch)
                if isinstance(batch_data, dict):
                    if 'fingerprints' in batch_data:
                        batch_fingerprints = batch_data['fingerprints']
                    elif 'fingerprints_packed' in batch_data:
                        from core.fingerprint import unpack_fingerprints
                        n_bits = batch_data.get('n_bits', 1024)
                        bitorder = batch_data.get('bitorder', 'little')
                        batch_fingerprints = unpack_fingerprints(
                            batch_data['fingerprints_packed'], n_bits, bitorder
                        )
                    else:
                        logger.error("No fingerprints found in batch data")
                        sys.exit(1)
                else:
                    batch_fingerprints = batch_data
                
                result = monitor.check_batch(batch_fingerprints)
                results.append(result)
                
                # Display results
                logger.info("\n" + "="*50)
                logger.info("DRIFT ANALYSIS RESULTS")
                logger.info("="*50)
                logger.info(f"Drift detected: {result['drift_detected']}")
                logger.info(f"Drift severity: {result['drift_severity']}")
                logger.info(f"JS divergence: {result['js_divergence']:.4f}")
                logger.info(f"KS statistic: {result['ks_statistic']:.4f}")
                logger.info(f"KS p-value: {result['ks_pvalue']:.4f}")
                logger.info(f"Recommend recalibration: {result['recommend_recalibration']}")
                
                if result['recommend_recalibration']:
                    logger.info(f"Reason: {result.get('recommendation_reason', 'Significant drift detected')}")
            
            else:
                # Interactive drift monitoring mode
                logger.info("Drift monitor initialized. Baseline statistics:")
                baseline_stats = monitor.baseline_stats
                logger.info(f"Baseline entropy: {baseline_stats['overall_entropy']:.4f}")
                logger.info(f"Mean bit activation: {np.mean(baseline_stats['bit_activation_rates']):.3f}")
                logger.info("\nUse --batch to analyze specific batch data")
            
            # Save drift analysis results
            if results:
                output_data = {
                    'baseline_stats': monitor.baseline_stats,
                    'drift_results': results,
                    'drift_history': monitor.get_drift_history(),
                    'drift_summary': monitor.get_drift_summary()
                }
                
                with open(args.output_drift, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                
                logger.info(f"\nDrift analysis saved to: {args.output_drift}")
        
        else:
            parser.error(f"Unknown mode: {args.mode}")
            
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()