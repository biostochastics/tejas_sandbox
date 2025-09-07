# DOE Framework Architecture

## Overview

The DOE (Design of Experiments) Framework is a comprehensive benchmarking system for systematically evaluating optimization strategies in binary semantic search. It reduces experimental complexity from 13,824 full factorial combinations to ~100 strategically selected experiments while maintaining statistical validity.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DOE Benchmark Framework                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Security & Reliability Layer (NEW)                                  │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ SafeEvaluator │ ResourceGuard │ Validators │ Utils         │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  Core Components                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   Factors    │  │   Encoders   │  │ Compatibility │             │
│  │   Registry   │  │   Factory    │  │  Validator   │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                  │                  │                      │
│         └──────────────────┼──────────────────┘                     │
│                            │                                         │
│  Experimental Pipeline     ▼                                         │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ Design Generator → Experiment Runner → DOE Analyzer        │     │
│  └────────────────────────────────────────────────────────────┘     │
│                            │                                         │
│  Output Layer              ▼                                         │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ Report Generator │ Visualizations │ Statistical Analysis   │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### Security & Reliability Layer (NEW)

#### SafeEvaluator (`core/safe_evaluator.py`)
- **Purpose**: Safe expression evaluation without eval()
- **Features**: AST-based parsing, whitelist of allowed operations
- **Security**: Prevents code injection attacks

#### ResourceGuard (`core/resource_guard.py`)
- **Purpose**: Resource limitation and DoS protection
- **Features**: 
  - Process isolation for experiments
  - Configurable timeout limits (default 300s)
  - Memory usage limits (default 2GB)
  - Automatic cleanup on limit exceeded
- **Security**: Prevents resource exhaustion attacks

#### Validators (`core/validators.py`)
- **Purpose**: Comprehensive input validation
- **Features**:
  - Factor value validation with bounds checking
  - DataFrame integrity validation
  - Configuration sanitization
  - Type conversion and coercion
- **Security**: Prevents invalid input exploitation

#### Utils (`core/utils.py`)
- **Purpose**: Defensive programming utilities
- **Features**:
  - `safe_divide()`: Division by zero protection
  - `validate_dataframe()`: Data structure validation
  - `ensure_finite()`: NaN/Inf handling
  - `sanitize_input()`: Input sanitization
- **Reliability**: Prevents crashes from edge cases

### Core Components

#### Factors Registry (`core/factors.py`)
- **Purpose**: Define and manage experimental factors
- **Features**:
  - 11 controllable factors (binary, categorical, continuous)
  - Constraint validation system
  - Factor compatibility rules
  - Configuration encoding/decoding

#### Encoder Factory (`core/encoder_factory.py`)
- **Purpose**: Safe encoder instantiation
- **Features**:
  - Registry-based encoder creation (no dynamic imports)
  - Configuration validation
  - Pipeline-specific optimizations
  - Clear error messages
- **Security**: Eliminates unsafe dynamic code loading

#### Compatibility Validator (`core/compatibility.py`)
- **Purpose**: Ensure valid factor combinations
- **Features**:
  - Hard/soft incompatibility rules
  - Automatic configuration fixing
  - SafeEvaluator integration
- **Security**: Uses SafeEvaluator instead of eval()

### Experimental Pipeline

#### Design Generator (`core/designs.py`)
- **Purpose**: Create experimental design matrices
- **Designs Supported**:
  - Plackett-Burman (screening)
  - Central Composite Design (optimization)
  - Box-Behnken Design
  - Latin Hypercube Sampling
  - Fractional Factorial
  - Custom designs

#### Experiment Runner (`run_doe_benchmark.py`)
- **Purpose**: Execute experiments safely
- **Features**:
  - Resource-limited execution via `run_single_experiment_safe()`
  - Parallel experiment support
  - Checkpoint/resume capability
  - Progress tracking and reporting
- **Security**: All experiments run with resource limits

#### DOE Analyzer (`core/doe_analysis.py`)
- **Purpose**: Statistical analysis of results
- **Features**:
  - Main effects analysis
  - Interaction detection
  - Response surface modeling
  - ANOVA analysis
  - Pareto frontier identification
- **Reliability**: Safe numerical operations throughout

### Output Layer

#### Report Generator
- **Purpose**: Generate comprehensive analysis reports
- **Output Formats**:
  - Markdown reports
  - JSON data exports
  - CSV result tables

#### Visualizations
- **Purpose**: Interactive data visualization
- **Technologies**: Plotly for interactive plots
- **Chart Types**:
  - Main effects plots
  - Interaction plots
  - Response surfaces
  - Pareto frontiers

## Data Flow

```
1. Configuration Input
   ↓ (Validation via Validators)
2. Design Generation
   ↓ (Compatibility checking)
3. Encoder Creation
   ↓ (Safe factory instantiation)
4. Experiment Execution
   ↓ (Resource-limited via ResourceGuard)
5. Data Collection
   ↓ (Safe numerical operations)
6. Statistical Analysis
   ↓ (Protected computations)
7. Report Generation
```

## Security Architecture

### Defense in Depth Strategy

1. **Input Layer**
   - All user inputs validated
   - Configuration sanitization
   - Bounds checking

2. **Execution Layer**
   - No eval() or exec() usage
   - No dynamic imports
   - Process isolation

3. **Resource Layer**
   - Memory limits enforced
   - Timeout protection
   - Graceful degradation

4. **Computation Layer**
   - Safe numerical operations
   - NaN propagation for undefined
   - Edge case handling

## Pipeline Architectures

| Pipeline | Security Features | Performance Features |
|----------|------------------|---------------------|
| original_tejas | Basic validation | Baseline |
| goldenratio | Full validation | Golden ratio optimization |
| fused_char | Process isolation | Fused operations |
| fused_byte | Resource limits | ByteBPE tokenization |
| optimized_fused | All security features | Maximum optimization |

## File Structure

```
benchmark_doe/
├── core/                      # Core framework components
│   ├── Security & Reliability (NEW)
│   │   ├── utils.py           # Defensive utilities
│   │   ├── resource_guard.py  # Resource protection
│   │   ├── validators.py      # Input validation
│   │   └── safe_evaluator.py  # Safe expression eval
│   │
│   ├── Experimental Core
│   │   ├── factors.py         # Factor definitions
│   │   ├── designs.py         # DOE designs
│   │   ├── doe_analysis.py    # Statistical analysis
│   │   └── encoder_factory.py # Safe encoder creation
│   │
│   └── Infrastructure
│       ├── compatibility.py   # Constraint validation
│       ├── dataset_loader.py  # Data loading
│       └── profiler.py        # Performance profiling
│
├── run_doe_benchmark.py       # Main execution script
├── test_fixes.py              # Security fix verification
├── FIXES_APPLIED.md          # Security fix documentation
├── README.md                  # User documentation
├── ARCHITECTURE.md           # This file
└── DOE_IMPLEMENTATION_GUIDELINES.md # Implementation guide
```

## Error Handling Strategy

1. **Fail-Safe Defaults**: Return NaN or safe values on error
2. **Graceful Degradation**: Continue with reduced functionality
3. **Comprehensive Logging**: Track all warnings and errors
4. **User-Friendly Messages**: Clear error descriptions

## Testing Strategy

- **Unit Tests**: Each utility function tested
- **Integration Tests**: End-to-end workflow validation
- **Security Tests**: Vulnerability scanning
- **Performance Tests**: Resource limit verification
- **Edge Case Tests**: Boundary condition handling

## Performance Considerations

- **Process Isolation Overhead**: ~50ms per experiment
- **Safe Division Overhead**: ~5% for numerical operations
- **Validation Overhead**: <1% for most operations
- **Memory Overhead**: Minimal due to efficient cleanup

## Future Enhancements

1. **Container-based Isolation**: Docker support for experiments
2. **Distributed Execution**: Multi-machine support
3. **Real-time Monitoring**: Live experiment dashboard
4. **Advanced Analytics**: Bayesian optimization integration
5. **Cloud Integration**: AWS/GCP/Azure support

## Security Compliance

- ✅ No eval() or exec() usage
- ✅ No unsafe dynamic imports
- ✅ Input validation on all entry points
- ✅ Resource limits enforced
- ✅ Safe numerical operations
- ✅ Process isolation for untrusted code

## Conclusion

The DOE Framework provides a robust, secure, and statistically valid approach to experimental design and analysis. With comprehensive security enhancements, it can safely handle untrusted input and resource-intensive operations while maintaining high performance and reliability.