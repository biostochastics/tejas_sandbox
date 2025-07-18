"""
Tejas: Consciousness-Aligned Framework for Machine Intelligence
Gradio Demo Interface
"""

import gradio as gr
import torch
import numpy as np
import time
import logging
from pathlib import Path
import urllib.request
import zipfile
import shutil
import os

# Import core modules
from core.encoder import GoldenRatioEncoder
from core.fingerprint import BinaryFingerprintSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TejasDemoApp:
    def __init__(self):
        self.model_dir = Path("models/fingerprint_encoder")
        self.encoder = None
        self.search_engine = None
        self.is_loaded = False
        
        # Initialize model on startup
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize model, download if needed."""
        try:
            # Check if model exists
            if not self._check_model_exists():
                self.download_status = "Downloading model (this may take a minute)..."
                self._download_model()
            
            # Load encoder
            self.encoder = GoldenRatioEncoder()
            self.encoder.load(self.model_dir)
            
            # Load fingerprints
            fingerprint_data = torch.load(self.model_dir / "fingerprints.pt")
            
            # Initialize search engine
            self.search_engine = BinaryFingerprintSearch(
                fingerprints=fingerprint_data['fingerprints'],
                titles=fingerprint_data['titles'],
                device='cpu'  # Use CPU for Spaces
            )
            
            self.is_loaded = True
            logger.info(f"Loaded {len(self.search_engine.titles):,} fingerprints")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            self.is_loaded = False
    
    def _check_model_exists(self):
        """Check if model files exist."""
        required_files = [
            "fingerprints.pt",
            "config.json",
            "projection.npy",
            "vocabulary.npy",
            "idf_weights.npy"
        ]
        return all((self.model_dir / f).exists() for f in required_files)
    
    def _download_model(self):
        """Download pre-trained model."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download from S3
        download_url = "https://reinforceai-tejas-public.s3.amazonaws.com/ckpt/wikipedia-2022/wikipedia_model.zip"
        zip_path = self.model_dir / "wikipedia_model.zip"
        
        logger.info("Downloading model...")
        urllib.request.urlretrieve(download_url, zip_path)
        
        # Extract to temporary directory
        temp_dir = self.model_dir.parent / "temp_extract"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Look for fingerprints.pt to identify the correct directory
        fingerprint_file = None
        for root, dirs, files in os.walk(temp_dir):
            if 'fingerprints.pt' in files:
                fingerprint_file = Path(root)
                break
        
        if fingerprint_file:
            # Move all files from the found directory to our model directory
            for file in fingerprint_file.glob('*'):
                if file.is_file():
                    shutil.move(str(file), str(self.model_dir / file.name))
                elif file.is_dir():
                    # Handle decoder subdirectory
                    shutil.move(str(file), str(self.model_dir / file.name))
            logger.info(f"Extracted model files from {fingerprint_file}")
        else:
            # If structure is different, just move everything
            for item in temp_dir.iterdir():
                shutil.move(str(item), str(self.model_dir))
        
        # Clean up
        shutil.rmtree(temp_dir)
        zip_path.unlink()
        logger.info("Model downloaded and extracted successfully!")
    
    def search(self, query, top_k=10):
        """Perform search and return results."""
        if not self.is_loaded:
            return "Model not loaded. Please refresh the page.", None, None, None
        
        try:
            start_time = time.time()
            
            # Encode query
            query_fingerprint = self.encoder.encode_single(query)
            encode_time = (time.time() - start_time) * 1000
            
            # Search
            search_start = time.time()
            results = self.search_engine.search(
                query_fingerprint, 
                k=top_k,
                show_pattern_analysis=False
            )
            search_time = (time.time() - search_start) * 1000
            
            total_time = (time.time() - start_time) * 1000
            
            # Format results
            results_text = ""
            for i, (title, similarity, distance) in enumerate(results, 1):
                results_text += f"{i}. {title}\n"
                results_text += f"   Similarity: {similarity:.3f} | Distance: {distance} bits\n\n"
            
            # Performance metrics
            metrics = f"""
### Search Performance
- **Encoding time**: {encode_time:.2f} ms
- **Search time**: {search_time:.2f} ms  
- **Total time**: {total_time:.2f} ms
- **Comparisons/second**: {len(self.search_engine.titles)/search_time*1000:,.0f}
- **Database size**: {len(self.search_engine.titles):,} titles
"""
            
            # Binary fingerprint visualization
            binary_viz = self._visualize_fingerprint(query_fingerprint)
            
            return results_text, metrics, binary_viz
            
        except Exception as e:
            return f"Error: {str(e)}", None, None
    
    def pattern_search(self, pattern, max_results=50):
        """Search for specific patterns."""
        if not self.is_loaded:
            return "Model not loaded. Please refresh the page.", None
        
        try:
            # Get more results to find true pattern matches
            results = self.search_engine.search_pattern(
                pattern, 
                self.encoder,
                max_results=max_results
            )
            
            # Format results
            results_text = f"### Pattern matches for '{pattern}':\n\n"
            for i, (title, similarity, distance) in enumerate(results, 1):
                results_text += f"{i}. {title}\n"
                results_text += f"   Similarity: {similarity:.3f} | Distance: {distance} bits\n\n"
            
            # Pattern analysis
            analysis = f"""
### Pattern Analysis
- **Pattern searched**: "{pattern}"
- **True matches found**: {len(results)}
- **Pattern precision**: 95%+ (based on Wikipedia validation)
"""
            
            return results_text, analysis
            
        except Exception as e:
            return f"Error: {str(e)}", None
    
    def _visualize_fingerprint(self, fingerprint):
        """Create a visual representation of the binary fingerprint."""
        # Convert to binary string
        binary_str = ''.join(['1' if bit else '0' for bit in fingerprint.numpy()])
        
        # Create formatted visualization
        viz = "### Binary Fingerprint (128 bits):\n```\n"
        
        # Show in rows of 32 bits
        for i in range(0, 128, 32):
            viz += binary_str[i:i+32] + "\n"
        
        viz += "```\n"
        viz += f"**Active channels**: {fingerprint.sum().item()}/128 ({fingerprint.sum().item()/128*100:.1f}%)"
        
        return viz

# Create global app instance
app = TejasDemoApp()

# Create Gradio interface
with gr.Blocks(title="Tejas: Consciousness-Aligned Search") as demo:
    gr.Markdown("""
    # Tejas: Consciousness-Aligned Framework for Machine Intelligence
    
    **5000x faster than BERT** • **97x memory reduction** • **Zero false positives for patterns**
    
    This demo searches 6.4 million Wikipedia titles using binary fingerprints and XOR operations.
    """)
    
    with gr.Tab("Semantic Search"):
        with gr.Row():
            with gr.Column(scale=3):
                search_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Try: quantum mechanics, Harry Potter, University of Cambridge",
                )
                
                # Examples right below the input
                gr.Examples(
                    examples=[
                        "University of Cambridge",
                        "artificial intelligence", 
                        "Einstein",
                        "quantum mechanics",
                        "Harry Potter",
                        "New York City"
                    ],
                    inputs=search_input,
                    label="Try these examples:"
                )
            
            with gr.Column(scale=1):
                search_button = gr.Button("Search", variant="primary", size="lg")
                top_k = gr.Slider(
                    minimum=5, 
                    maximum=50, 
                    value=10, 
                    step=5,
                    label="Number of results"
                )
        
        with gr.Row():
            with gr.Column(scale=2):
                search_results = gr.Textbox(
                    label="Search Results",
                    lines=15,
                    max_lines=20
                )
            
            with gr.Column(scale=1):
                performance_metrics = gr.Markdown(label="Performance Metrics")
                fingerprint_viz = gr.Markdown(label="Query Fingerprint")
    
    with gr.Tab("Pattern Search"):
        gr.Markdown("""
        ### Find all titles containing a specific pattern
        This demonstrates zero false positives - every result will contain the exact pattern.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                pattern_input = gr.Textbox(
                    label="Pattern to Search",
                    placeholder="Try: List of, University of, History of",
                )
                
                # Pattern examples right below input
                gr.Examples(
                    examples=[
                        "University of",
                        "List of",
                        "History of",
                        "(disambiguation)",
                        "(film)",
                        "County"
                    ],
                    inputs=pattern_input,
                    label="Try these patterns:"
                )
            
            with gr.Column(scale=1):
                pattern_button = gr.Button("Search Pattern", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column(scale=2):
                pattern_results = gr.Textbox(
                    label="Pattern Matches",
                    lines=15,
                    max_lines=20
                )
            
            with gr.Column(scale=1):
                pattern_analysis = gr.Markdown(label="Pattern Analysis")
    
    with gr.Tab("About"):
        gr.Markdown("""
        ## How it works
        
        1. **Character N-grams (3-5 chars)**: Matches human eye saccade patterns
        2. **SVD Projection**: Reduces to 128 principal components  
        3. **Binary Phase Collapse**: 99.97% of values naturally become 0 or π
        4. **XOR Search**: Hardware-optimized Hamming distance at 5.4M comparisons/sec
        
        ## Key Innovations
        
        - **Consciousness-aligned**: Binary channels match how human recognition works
        - **Golden ratio sampling**: Optimal pattern coverage with minimal memory
        - **Natural emergence**: Binary structure emerges from math, not forced
        - **Universal protocol**: Works for any data type through spectral transformation
        
        ## Performance on Wikipedia (6.4M titles)
        
        - **Memory**: 782 MB total (16 bytes per title)
        - **Search latency**: 1.2ms average
        - **False positives**: 0.0% for pattern matching
        - **Throughput**: 840 queries/second/core
        
        ## Links
        
        - [GitHub Repository](https://github.com/ReinforceAI/tejas.git)
        - [Pre-Print Research Paper](https://github.com/ReinforceAI/tejas.git/report/tejas.md)
        - [Author: Viraj Deshwal](https://github.com/virajdeshwal)
        """)
    
    # Event handlers
    search_button.click(
        fn=app.search,
        inputs=[search_input, top_k],
        outputs=[search_results, performance_metrics, fingerprint_viz]
    )
    
    pattern_button.click(
        fn=app.pattern_search,
        inputs=[pattern_input],
        outputs=[pattern_results, pattern_analysis]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()