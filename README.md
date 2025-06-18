# Fast-DLLM
[![Project](https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages)](https://nvlabs.github.io/Fast-dLLM)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2505.22618)

Fast-DLLM is a diffusion-based Large Language Model (LLM) inference acceleration framework that supports efficient inference for models like Dream and LLaDA.

<div align="center">
  <img src="asset/speedup.jpg" alt="End-to-end speedup over vanilla LLaDA baseline" width="800"/>
  <p>End-to-end speedup over vanilla LLaDA baseline</p>
</div>

## Project Structure

```
.
├── dream/          # Dream model related code
├── llada/          # LLaDA model related code
└── .gitignore      # Git ignore configuration
```

## Features

- Fast inference support for Dream and LLaDA models
- Multiple inference optimization strategies
- Code generation and evaluation capabilities
- Interactive chat interface

### Key Features

1. **Key-Value Cache for Block-Wise Decoding**
   We propose an efficient block-wise decoding KV Cache mechanism for Masked Diffusion Models (MDMs). By reusing attention Key-Value activations across multiple steps within each block, our approach avoids redundant computation and significantly accelerates inference. Furthermore, our DualCache extension also caches masked suffix tokens, enabling even greater speedup with negligible accuracy loss.

<div align="center">
  <img src="asset/kvcache.jpg" alt="KV Cache for block-wise decoding" width="800"/>
  <p>KV Cache for block-wise decoding</p>
</div>

2. **Confidence-Aware Parallel Decoding**
   Instead of decoding tokens sequentially, we introduce a confidence-aware parallel decoding scheme. At each step, only tokens with confidence over a threshold are unmasked in parallel, while uncertain ones remain masked for future steps. This selective approach effectively balances decoding efficiency and output quality.

<div align="center">
  <img src="asset/output.gif" alt="Decoding comparison" width="800"/>
  <p><b>Left:</b> Standard decoding (LLaDA). <b>Right:</b> Confidence-aware parallel decoding.</p>
</div>

<div align="center">
  <img src="asset/pseudo_code.jpg" alt="Pseudo code for our method" width="800"/>
  <p>Pseudo code for our method</p>
</div>

3. **Overall Performance**
   Overall, introducing the KV Cache mechanism yields significant speed improvements for all tasks and sequence lengths, typically achieving a 2x to 3.6x speedup compared to the vanilla backbone. When the parallel decoding strategy is applied individually, we see additional acceleration, often pushing speedups to 4x-6x for the evaluated settings, particularly as the generation length increases.

<div align="center">
  <img src="asset/overall_performance.jpg" alt="Overall performance" width="800"/>
  <p>Overall performance comparison</p>
</div>

4. **Dynamic Refinement Steps (DRS)**
   We introduce Dynamic Refinement Steps (DRS), an advanced extension that adaptively allocates computational budget based on block-level confidence analysis. DRS achieves remarkable efficiency gains while maintaining generation quality through intelligent refinement targeting.

   **Key Results:**
   - **50% NFE Reduction**: Average 0.5x compute compared to baseline
   - **Quality Maintenance**: Minimal quality degradation (-0.007 average)  
   - **77.8% Success Rate**: Excellent or Good performance across diverse tasks
   - **Optimal Configuration**: Aggressive settings (t_base=4, threshold=0.7) achieve best trade-offs

   | Configuration | Success Rate | NFE Ratio | Quality Change |
   |---------------|--------------|-----------|----------------|
   | DRS-Aggressive | 100% (3/3) | 0.401x | +0.009 |
   | DRS-Balanced | 66.7% (2/3) | 0.529x | -0.010 |
   | DRS-Conservative | 66.7% (2/3) | 0.565x | -0.020 |

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/fast-dllm.git
cd fast-dllm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Using LLaDA Model

#### Interactive Chat
```bash
python llada/chat.py --gen_length 128 --steps 128 --block_size 32
```

#### Dynamic Refinement Steps (DRS)
```bash
# Standard DRS with optimal settings
python llada/generate.py --use_drs --t_base 4 --threshold 0.7

# Run DRS validation experiments
python llada/test_drs.py
```

Parameter descriptions:
- `--gen_length`: Maximum length of generated text
- `--steps`: Number of sampling steps
- `--block_size`: Cache block size
- `--use_cache`: Whether to use cache
- `--if_cache_position`: Whether to use dual cache
- `--threshold`: Confidence threshold for DRS
- `--t_base`: Base steps per block in DRS
- `--use_drs`: Enable Dynamic Refinement Steps

#### Web Demo
We also provide a web demo using Gradio. First, install Gradio:
```bash
pip install gradio
```

Then run the demo:
```bash
cd llada
python app.py
```

#### Model Evaluation
| Benchmark         | Gen Length | LLaDA   | +Cache         | +Parallel      | +Cache+Parallel (Fast-dLLM) |
|-------------------|------------|---------|----------------|----------------|-----------------------------|
| **GSM8K (5-shot)**| 256        | 79.3<br>6.73<br>(1×) | 79.5<br>21.23<br>(3.2×) | 79.2<br>16.53<br>(2.5×) | 78.5<br>**54.4<br>(8.1×)** |
|                   | 512        | 77.5<br>3.23<br>(1×) | 77.0<br>10.43<br>(3.3×) | 77.6<br>18.63<br>(5.8×) | 77.2<br>**35.3<br>(11.0×)** |
| **HumanEval (0-shot)** | 256   | 41.5<br>30.5 (1×) | 42.7<br>40.73<br>(1.3×) | 43.9<br>101.53<br>(3.3×) | 43.3<br>**114.1<br>(3.7×)** |
|                   | 512        | 43.9<br>18.4 (1×) | 45.7<br>29.33<br>(1.6×) | 43.3<br>57.13<br>(3.1×) | 44.5<br>**73.7<br>(4.0×)** |

Each cell presents the accuracy (top row, in percentage) and the decoding throughput (middle row, in tokens per second) with relative speedup (bottom row) to the LLaDA baseline.

#### Dynamic Refinement Steps (DRS) Results

| Task Type | DRS-Aggressive | DRS-Balanced | DRS-Conservative |
|-----------|----------------|--------------|------------------|
| **Math (Rectangular Area)** | ✅ GOOD<br>0.27× NFE<br>Quality: +0.014 | ✅ GOOD<br>0.42× NFE<br>Quality: -0.004 | ✅ GOOD<br>0.51× NFE<br>Quality: -0.016 |
| **Code (Factorial)** | ✅ GOOD<br>0.31× NFE<br>Quality: +0.017 | ❌ POOR<br>0.51× NFE<br>Quality: -0.021 | ❌ POOR<br>0.52× NFE<br>Quality: -0.052 |
| **Explanation (Photosynthesis)** | 🌟 EXCELLENT<br>0.62× NFE<br>Quality: +0.024 | ✅ GOOD<br>0.66× NFE<br>Quality: -0.005 | ✅ GOOD<br>0.66× NFE<br>Quality: +0.007 |

**DRS achieves up to 73% NFE reduction while maintaining or improving generation quality, with DRS-Aggressive showing the best overall performance.**

For detailed evaluation instructions on GSM8K and HumanEval benchmarks, please refer to [LLaDA Evaluation Guide](llada/eval.md).

### 2. Using Dream Model

For detailed evaluation instructions on GSM8K and HumanEval benchmarks, please refer to [Dream Evaluation Guide](dream/eval.md).

## Contributing

Issues and Pull Requests are welcome!

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. 

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{wu2025fastdllmtrainingfreeaccelerationdiffusion,
      title={Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding}, 
      author={Chengyue Wu and Hao Zhang and Shuchen Xue and Zhijian Liu and Shizhe Diao and Ligeng Zhu and Ping Luo and Song Han and Enze Xie},
      year={2025},
      eprint={2505.22618},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.22618}, 
}
```

## Acknowledgements

We would like to thank the authors of [LLaDA](https://github.com/llada-project/llada) and [Dream](https://github.com/dream-project/dream) for their excellent work and open-source contributions. 