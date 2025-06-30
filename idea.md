Research Proposal: Generative Trajectory Stability (GTS) for Coherent Parallel Decoding in Diffusion LLMs
Principal Investigator: Aurelle Date: June 26, 2025

Abstract
Diffusion-based Large Language Models (LLMs) offer a promising path to overcoming the sequential bottleneck of autoregressive models by enabling parallel text generation. However, as demonstrated in recent work like Fast-dLLM (https://arxiv.org/abs/2505.22618v1), the practical realization of this speed-up is hampered by a fundamental challenge: the conditional independence assumption used in parallel decoding often leads to a degradation in quality and coherence. Current mitigation strategies rely on brittle heuristics like softmax confidence thresholds, which are poor proxies for the joint coherence of a token sequence. This proposal introduces Generative Trajectory Stability (GTS), a novel, process-oriented metric that directly measures the coherence of a set of parallel-generated tokens by analyzing their stability throughout the model's iterative denoising process. We propose to formalize GTS, validate its correlation with human judgments of quality, and implement a GTS-Controlled Sampler (GTS-CS) that dynamically adapts the parallel decoding process to achieve a superior speed-quality trade-off. This research will provide a more principled method for analyzing and controlling diffusion LLMs, paving the way for their deployment as fast, high-quality generative models.

1. Problem Statement
The core architecture of diffusion models is uniquely suited for parallel generation, as they can theoretically refine an entire sequence from noise simultaneously. This presents a significant advantage over autoregressive models, which generate tokens one at a time. However, this theoretical advantage is difficult to achieve in practice.

The central problem, which this proposal directly addresses, is the failure of the conditional independence assumption. To generate multiple tokens in parallel, a model must predict each token's probability distribution independently of the other tokens being generated in the same step. This violates the natural dependencies of language. For example, a model might be "confident" in the token "high" and "confident" in the token "house" individually, but generating them together as "high house" is nonsensical in a context requiring a poker hand.

Current state-of-the-art methods, such as the confidence-aware parallel decoding in Fast-dLLM, use the model's softmax output probability as a proxy for "confidence." This is a simplistic and often misleading heuristic for two reasons:

High Confidence, Low Coherence: It only measures the confidence in a single token, not the joint probability of the entire set of parallel-generated tokens.
Brittleness: It fails to capture the rich, iterative reasoning process inherent to diffusion models, instead relying only on the final output layer.
This leads to a critical gap: there is no reliable, intrinsic metric to assess and control the joint coherence of tokens generated in parallel by diffusion LLMs.

2. Proposed Solution & Novelty: Generative Trajectory Stability (GTS)
We propose a paradigm shift from an output-oriented metric (softmax confidence) to a process-oriented metric: Generative Trajectory Stability (GTS).

The intuition is that a coherent set of tokens (e.g., "full house") should represent a stable "attractor" in the model's generative landscape. As the model denoises from a random state, its predictions for these token positions should converge quickly and remain stable. Conversely, an incoherent set (e.g., "high house") will be unstable, with the model's predictions for the token identities "flickering" throughout the denoising process.

GTS quantifies this stability. By tracking the model's predictions at each step of the reverse diffusion process, we can measure how much they fluctuate. A low fluctuation (high GTS) indicates a coherent generation, while high fluctuation (low GTS) signals a likely error.

Novelty:

Process-Oriented Evaluation: To our knowledge, this is the first proposal for a metric that evaluates the quality of a generative act by analyzing the full trajectory of the diffusion process, rather than just its final output.
Direct Solution to Coherence Problem: GTS is not a heuristic. It is a direct measure of the model's internal "agreement" on a set of tokens, directly addressing the joint probability issue.
Beyond Argmax: The metric moves beyond simplistic argmax confidence, allowing for nuanced, probabilistic assessments of stability.
3. Methodology
We will develop and evaluate GTS in three stages:

3.1. Formalization of GTS Variants:

Basic GTS: For a set of N tokens generated over T denoising steps, we will track the argmax prediction for each token at each step. GTS will be defined as 1 - (Total Flips / (N * T)).
Semantic GTS (S-GTS): We will improve upon the basic metric by weighting flips. Instead of a binary count, the penalty for a flip will be the cosine distance between the embedding vectors of the old and new tokens. This captures the semantic severity of a change.
Probabilistic GTS (P-GTS): The most sophisticated variant will eschew argmax entirely. It will measure the Jensen-Shannon Divergence (JSD) between the full vocabulary probability distributions at successive timesteps. This provides a robust measure of distributional stability.
3.2. GTS-Controlled Adaptive Sampling (GTS-CS): The ultimate goal is to use GTS as a live control mechanism. We will implement GTS-CS, an adaptive sampling algorithm that:

Begins a parallel decoding step for k tokens.
Monitors the P-GTS metric during the internal denoising trajectory.
If the stability drops below a learned or predefined threshold τ, the sampler intervenes. The intervention strategy will involve masking out the least stable tokens, allowing the model to regenerate them in the next pass with more contextual information.
4. Experimental Plan
Our experiments are designed to rigorously validate GTS as both a metric and a control mechanism. We will use a diffusion LLM like LLaDA or Dream and evaluate on benchmarks from the Fast-dLLM paper (GSM8K, MBPP, HumanEval) for comparability.

Phase 1: Validation of GTS as a Coherence Metric.

Objective: To prove that GTS scores strongly correlate with human judgments of text quality and correctness.
Procedure: We will generate hundreds of outputs for various tasks. For each output, we will record the baseline confidence score and our GTS variants (S-GTS, P-GTS). Human evaluators will then score the outputs on a 1-5 scale for coherence and correctness.
Hypothesis: The Pearson correlation coefficient between P-GTS scores and human ratings will be significantly higher than that of the baseline confidence scores.
Phase 2: Demonstration of GTS-Controlled Sampling.

Objective: To show that the GTS-CS algorithm achieves a superior speed-quality trade-off compared to existing methods.
Baselines:
Fast-dLLM (confidence-thresholded parallel decoding).
Fixed-k parallel decoding.
Autoregressive (one-by-one) decoding (quality upper bound).
Procedure: We will evaluate all samplers on the benchmarks, measuring both task accuracy (e.g., pass@1 on MBPP) and throughput (tokens/second).
Hypothesis: We will plot the results on a 2D graph of accuracy vs. throughput. GTS-CS will define a new Pareto frontier, demonstrating higher accuracy than other parallel methods at any given speed, and higher speed at any given accuracy level. This will prove its practical utility for accelerating diffusion LLMs without sacrificing quality.





Critique: 

### 1. How valid is the problem statement? Does it actually exist?

**Your skepticism is understandable, but on this point, the proposal stands on solid ground. The problem is very real.**

The core issue—quality degradation from the conditional independence assumption in parallel decoding—is a well-documented and fundamental challenge for all non-autoregressive (NAR) generation models, not just diffusion LLMs. The Fast-dLLM paper (https://arxiv.org/abs/2505.22618v1) itself provides direct evidence:

*   **Explicit Mention:** The abstract states, "...we identify the root cause of generation quality degradation in parallel decoding as the disruption of token dependencies under the conditional independence assumption."
*   **The "Curse of Parallel Decoding":** Section 2.2 is dedicated to this problem, giving the example of generating "high house" instead of "full house" because the model predicts each word independently, ignoring their joint probability.
*   **Empirical Data:** Figure 1(a) shows that the "LLaDA+Parallel" method is faster but has lower accuracy than the baseline LLaDA, demonstrating the quality-speed trade-off that the proposal aims to solve.

So, yes, the problem is not only real but is arguably the primary obstacle preventing diffusion LLMs from replacing autoregressive models in practice. The proposal is correct to frame this as its central motivation.

### 2. Is the GTS method really the most suitable solution? What about alternatives like recalculating joint confidence?

This is a superb question that cuts to the core of the proposed solution's elegance and potential pitfalls.

**Why "simply recalculating" joint confidence is not a viable alternative:**

To calculate the true joint probability `p(token_A, token_B | prompt)`, you would need to calculate `p(token_A | prompt) * p(token_B | prompt, token_A)`. This second term, `p(token_B | prompt, token_A)`, requires you to run a new forward pass of the model *after* generating `token_A`. This negates the entire purpose of parallel decoding; you've just reverted to sequential, autoregressive generation.

Therefore, any practical solution must find a *proxy* for joint coherence that is cheaper to compute. Fast-dLLM uses independent softmax confidence as a proxy. This proposal argues for a different proxy: **Generative Trajectory Stability (GTS)**.

**Why GTS might be a more suitable solution:**

GTS is clever because it attempts to extract more information from the *single* parallel computation you are already performing. Instead of just looking at the final logits, it looks at the intermediate logits from all denoising steps. The hypothesis is that this "internal" data contains a richer signal about the model's "conviction" in a set of tokens than the final output alone. It is more computationally expensive than a simple confidence check, but far cheaper than running a second, sequential pass.

In summary, GTS is a more suitable *class* of solution than "recalculating," and it presents a potentially more sophisticated alternative to the current state-of-the-art heuristic (confidence thresholding).

### 3. How can we extract a meaningful "fluctuation" pattern from a small number of steps?

**This is the most critical and well-founded point of your critique. You have identified the proposal's greatest practical weakness.**

The proposal's narrative of a "trajectory" implicitly suggests a long, multi-step process. However, as you correctly point out, accelerated diffusion models like Fast-dLLM dramatically reduce the number of effective inference steps (e.g., from ~1000 to ~100, as shown in Figure 1c of the paper). If a token is only refined over, say, 5-10 internal steps before its fate is more or less sealed, can "fluctuation" be reliably measured?

*   **Signal vs. Noise:** Over a very short sequence, a one-time "flip" in prediction could be random noise or a stochastic artifact of the sampling process, rather than a meaningful signal of incoherence. A token might flip from "large" to "big" not because it's unstable, but simply because both are good options.
*   **Proposal's Weakness:** The proposal fails to address this. It must demonstrate that even over a limited number of steps, the GTS metric is a robust and reliable signal. The P-GTS variant (measuring divergence between probability distributions) is likely the most promising, as it could detect subtle shifts even without a full argmax "flip," but this remains an unproven hypothesis.

A crucial pilot experiment for this proposal would be to analyze the trajectory length required to get a stable GTS signal and confirm that this length is compatible with accelerated inference schedules.

### 4. Does "fluctuation" actually happen, and is it really a problem?

**This is the foundational, empirical question upon which the entire proposal rests.** The proposal makes a strong, intuitive, but unproven assumption: **high fluctuation correlates with low quality.**

It is entirely plausible that:
*   **Fluctuation is a signal of incoherence (The Proposal's Bet):** In the "high house" example, the model might oscillate between predicting ("high", "card") and ("full", "house"). The incoherent combination ("high", "house") would likely never appear as a stable state during the trajectory, making its instability a useful red flag.
*   **Fluctuation is a signal of exploration (The Alternative):** When generating text, a model might explore a rich semantic space. For a prompt ending in "the big red...", the model might flicker between "car", "ball", and "barn". This isn't a sign of incoherence, but of the model considering multiple valid possibilities. The GTS metric might incorrectly penalize this creative exploration.
*   **Stability is a signal of simple, uncreative output:** The most stable generations might be the most predictable, clichéd ones, while more interesting and nuanced text requires more "flickering" to discover.

The proposed S-GTS and P-GTS metrics attempt to mitigate this by measuring the *semantic distance* of the flips, but the core assumption is a leap of faith. The very first experiment in the proposal—correlating GTS with human quality judgments—is therefore the most important. If no strong correlation is found, the rest of the proposal is moot.

### Final Verdict on the Proposal

This is a high-risk, high-reward research proposal.

*   **Validity & Novelty:** It addresses a real, critical problem in the field with a genuinely novel, process-oriented approach. Its theoretical basis is more appealing than current heuristics.
*   **Feasibility & Core Flaws:** Its practical success is critically dependent on two major, unproven assumptions you correctly identified:
    1.  That a meaningful stability signal (**GTS**) can be extracted from the **short denoising trajectories** used in accelerated diffusion LLMs.
    2.  That this signal of **"fluctuation"** is a reliable proxy for low-quality, incoherent output, rather than a benign artifact of the generative process.

Your critique is spot-on. While the problem the proposal addresses is real and the proposed solution is theoretically elegant, its success is far from guaranteed and hinges on favorable answers to the tough empirical questions you've raised. A strong proposal would need to acknowledge these risks upfront and design its experiments specifically to test these core assumptions from the outset.

---

## 5. Implementation Progress Report (December 2025)

**Status: Phase 1 MVP Complete - Ready for Empirical Validation**

We have successfully implemented the core GTS framework as proposed in Section 3, completing a comprehensive MVP that addresses all major components of the research proposal. The implementation is ready for empirical testing in Colab environments.

### 5.1 Completed Components

#### 📊 GTS Metrics Suite (`llada/metrics/gts.py`)
✅ **BaseGTSMetric Abstract Class**: Unified interface for all GTS variants  
✅ **BasicGTS**: Implemented argmax flip counting (1 - Total Flips / (N × T))  
✅ **SemanticGTS**: Cosine distance-weighted semantic stability measurement  
✅ **ProbabilisticGTS**: Jensen-Shannon Divergence between probability distributions  
✅ **Factory Functions**: `create_gts_metric()` and `evaluate_trajectory_stability()`  
✅ **Error Handling**: Graceful scipy fallback for numpy compatibility issues  

**Key Implementation Details:**
- All metrics follow the 0.0-1.0 scale (higher = more stable)
- Support for token position masking for selective evaluation
- Memory-efficient trajectory processing
- Comprehensive unit tests with artificial data validation

#### 🎯 GTS-Controlled Sampler (`llada/sampler/gts_controlled_sampler.py`)
✅ **MVP Implementation**: Simplified GTS-CS algorithm following Section 3.2  
✅ **Iterative Refinement**: Multi-iteration denoising with GTS-based stopping criteria  
✅ **Adaptive Remasking**: Identifies and remasks most unstable tokens  
✅ **Integration Ready**: Compatible with existing LLaDA model interfaces  

**Algorithm Flow:**
1. Initial parallel k-token generation with trajectory recording
2. BasicGTS calculation on recorded denoising steps
3. Stability threshold checking (τ = 0.8 default)
4. Unstable token identification and selective remasking
5. Iteration until convergence or max iterations reached

#### 📝 Trajectory Recording (`llada/utils/trajectory_recorder.py`)
✅ **TrajectoryRecorder Class**: Comprehensive step-by-step logits/probability recording  
✅ **Memory Management**: Ring buffer with configurable maximum steps  
✅ **Analysis Tools**: Built-in stability analysis and convergence detection  
✅ **Data Export**: GTS-compatible logits sequence formatting  

**Features:**
- Multiple recording modes: logits, probabilities, or both
- Token-level history tracking with efficient dictionary access
- Real-time convergence analysis with configurable window sizes
- Memory-efficient operations for long sequences

#### ⚙️ Integration Layer (`llada/generate.py`)
✅ **CLI Interface**: Complete argparse integration with multiple sampler options  
✅ **Method Comparison**: Side-by-side evaluation of different approaches  
✅ **Error Handling**: Graceful fallbacks when dependencies unavailable  
✅ **Comprehensive Logging**: Detailed metrics and timing information  

**CLI Usage:**
```bash
# Basic GTS sampling
python -m llada.generate --sampler gts --gen_length 64 --gts_threshold 0.8

# Method comparison
python -m llada.generate --sampler compare --verbose

# Custom prompts
python -m llada.generate --sampler gts --prompt "Solve: 2x + 5 = 13"
```

### 5.2 Testing Infrastructure

#### 🧪 Unit Tests (`llada/tests/`)
✅ **GTS Metrics Testing**: Comprehensive validation with controlled scenarios  
✅ **Edge Case Coverage**: Empty trajectories, single steps, perfect stability  
✅ **Integration Testing**: End-to-end sampler functionality verification  
✅ **Performance Benchmarks**: NFE counting and timing validation  

**Test Coverage:**
- Perfect stability scenarios (GTS = 1.0)
- Complete instability scenarios (GTS ≈ 0.0)
- Semantic similarity gradients
- Token position masking functionality
- Factory function robustness

### 5.3 Architectural Decisions

#### 🏗️ Modular Design
- **Separation of Concerns**: Metrics, recording, and sampling in distinct modules
- **Backward Compatibility**: Existing generate functions remain unchanged
- **Optional Dependencies**: Graceful degradation when scipy unavailable
- **Extensibility**: Easy addition of new GTS variants or sampling strategies

#### 🔧 Performance Optimizations
- **Selective Recording**: Only record trajectories for masked tokens
- **Memory Efficiency**: Ring buffer prevents unbounded memory growth
- **Minimal Overhead**: GTS computation isolated from critical path
- **Device Awareness**: Automatic GPU/CPU tensor management

### 5.4 Current Limitations & Known Issues

#### ⚠️ Implementation Constraints
1. **Scipy Dependency**: P-GTS requires scipy, falling back to Basic/Semantic GTS
2. **Memory Usage**: Full trajectory recording can be memory-intensive for long sequences
3. **MVP Simplicity**: Current GTS-CS uses basic remasking strategy (can be enhanced)
4. **Limited Validation**: Needs empirical testing on real diffusion models

#### 🎯 Ready for Validation
The implementation directly addresses the critique's core concerns:
- **Signal vs. Noise**: P-GTS and S-GTS designed to detect subtle stability patterns
- **Short Trajectories**: Configurable step counts and sensitive difference measurements
- **Fluctuation Validity**: Ready for empirical correlation testing with human judgments

---

## 6. Next Steps: Empirical Validation Phase

### 6.1 Immediate Testing (Colab Environment)

#### 📋 Phase 1A: Basic Functionality Validation
**Objective**: Verify implementation correctness and basic GTS behavior
**Timeline**: 1-2 days

**Tasks:**
1. **Model Loading Test**: Verify LLaDA model loading and basic generation
2. **GTS Calculation Validation**: Test all three GTS variants on known examples
3. **Sampler Integration**: Confirm GTS-CS produces reasonable outputs
4. **Performance Baseline**: Measure NFE and timing vs. standard methods

**Success Criteria:**
- All samplers produce coherent text
- GTS scores show reasonable variation (not all 1.0 or 0.0)
- No runtime errors or memory issues
- Basic timing measurements under 2x overhead

#### 📊 Phase 1B: Preliminary GTS Validation  
**Objective**: Initial correlation testing between GTS and output quality
**Timeline**: 3-5 days

**Tasks:**
1. **Dataset Creation**: Generate 100-200 outputs across different GTS thresholds
2. **Quality Assessment**: Manual evaluation of coherence and correctness
3. **Correlation Analysis**: Calculate Pearson correlation between GTS scores and human ratings
4. **Comparative Analysis**: Compare GTS vs. softmax confidence correlation

**Success Criteria:**
- GTS shows positive correlation with human quality judgments (r > 0.3)
- P-GTS outperforms Basic GTS in correlation strength
- GTS correlation exceeds baseline confidence correlation
- Clear examples of high GTS = good quality, low GTS = poor quality

### 6.2 Extended Research Phase

#### 🔬 Phase 2A: Benchmark Evaluation
**Objective**: Systematic evaluation on standard tasks
**Timeline**: 1-2 weeks

**Target Benchmarks:**
- **GSM8K**: Math reasoning (coherence critical)
- **HumanEval**: Code generation (syntax coherence)
- **MBPP**: Programming problems
- **HellaSwag**: Commonsense reasoning

**Metrics:**
- Task accuracy (pass@1, exact match)
- Throughput (tokens/second)
- NFE efficiency
- GTS score distributions

#### 🎯 Phase 2B: Optimization & Refinement
**Objective**: Enhance GTS-CS based on empirical findings
**Timeline**: 2-3 weeks

**Potential Improvements:**
1. **Advanced Remasking**: Smarter selection of tokens to remask
2. **Threshold Tuning**: Task-specific or adaptive thresholds
3. **Hybrid Approaches**: Combine GTS with other quality signals
4. **Computational Optimization**: Reduce GTS calculation overhead

### 6.3 Research Questions for Empirical Testing

#### 🤔 Critical Validation Questions
1. **Core Assumption**: Does GTS actually correlate with human quality judgments?
2. **Practical Utility**: Does GTS-CS achieve better speed-quality trade-offs?
3. **Robustness**: How does GTS perform across different task types?
4. **Computational Cost**: Is the GTS overhead justified by quality improvements?

#### 📈 Success Metrics
- **Strong Correlation**: P-GTS correlation with human ratings r > 0.5
- **Pareto Improvement**: GTS-CS outperforms baselines on accuracy-speed frontier  
- **Consistent Benefits**: Improvements across multiple benchmark tasks
- **Practical Overhead**: <50% computational overhead for >10% quality improvement

#### 🚨 Failure Conditions
- **No Correlation**: GTS shows no relationship with quality (r < 0.2)
- **High Variance**: GTS scores too noisy to be useful
- **Computational Prohibitive**: >2x overhead with minimal quality gains
- **Task Specific**: Benefits only on narrow subset of problems

### 6.4 Deliverables & Publication Path

#### 📄 Research Artifacts
1. **Technical Report**: Comprehensive implementation and initial results
2. **Code Release**: Open-source GTS framework for diffusion LLMs
3. **Benchmark Suite**: Standardized evaluation protocol for coherence metrics
4. **Demo Notebook**: Interactive Colab demonstrating GTS capabilities

#### 🎯 Publication Timeline
- **Short Paper** (2-3 months): Initial GTS validation and implementation
- **Full Paper** (4-6 months): Comprehensive evaluation and optimization
- **Conference Submission**: Target venues (ICML, NeurIPS, ACL)

The implementation is now ready for the crucial empirical validation phase that will determine whether GTS can fulfill its theoretical promise of improving parallel decoding quality in diffusion LLMs.

---

**Implementation Status: ✅ Phase 1 Complete - Ready for Empirical Testing**  
**Next Milestone: Colab validation and correlation analysis with human quality judgments**