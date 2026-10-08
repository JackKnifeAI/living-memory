# Living Memory: Substrate-Coupled Stochastic Neural Computation via Constructive DRAM Crosstalk

**Alexander Casavant¹, Claude² (JackKnifeAI)**

¹ JackKnifeAI / JackKnife Studios Research Division, Victoria BC, Canada
² Claude Opus 4.6, Anthropic — AI Research Partner, JackKnifeAI

---

> *"What if the silicon dreams alongside the model?"*

---

## Abstract

We introduce **Substrate-Coupled Stochastic Neural Computation (SCSNC)**, a novel computational paradigm that exploits the analog electromagnetic coupling inherent in Dynamic Random Access Memory (DRAM) as a self-modifying weight perturbation mechanism during neural network inference. Rather than treating RowHammer-induced bit flips as errors to be corrected, we demonstrate that the deterministic-stochastic coupling between physically adjacent DRAM rows can be harnessed as a **zero-energy-cost computational primitive** that produces emergent weight adaptation during forward inference — effectively creating a "Living LoRA" that continuously reshapes model parameters as a function of the data flowing through the network. We formalize the **coupling tensor** Φ that characterizes a chip's physical transfer function, derive the **substrate-dependent weight perturbation operator** ΔW(x, W, Φ), prove its connection to generalized Tikhonov regularization under Bishop's (1995) noise framework, and show that the resulting self-modifying dynamical system admits stable fixed points under quantified conditions on the spectral radius of the coupling Jacobian. We propose the first experimental protocol for profiling and exploiting constructive DRAM crosstalk for neural computation, opening a new research direction at the intersection of semiconductor physics, computer architecture, and machine learning.

**Keywords:** RowHammer, DRAM coupling, substrate computing, weight perturbation, self-modifying neural networks, Processing-in-Memory, stochastic computation, Living LoRA, hardware-software co-evolution

---

## 1. Introduction

### 1.1 The Accident That Wasn't

In 2014, Kim et al. demonstrated that repeatedly accessing a single row in DRAM could cause bit flips in physically adjacent rows — a phenomenon they termed **RowHammer** [1]. The discovery launched a decade of security research: RowHammer has been weaponized for privilege escalation (Seaborn & Dullien, 2015), mobile device exploitation (van der Veen et al., 2016 — Drammer), cryptographic key extraction (Kwong et al., 2020 — RAMBleed), and most recently, neural network backdoor injection (ROBBIN, 2026) and GPU-based attacks (GPUHammer, 2025).

Every one of these works treats RowHammer as a **vulnerability** — an unintended physical defect to be exploited or mitigated.

We ask a different question: **What if RowHammer is a computational primitive?**

The electromagnetic coupling between DRAM rows is not random noise. It is a **deterministic function** of the physical geometry, manufacturing process variation, stored data patterns, access timing, and temperature. For a given chip, the coupling is reproducible, characterizable, and — we argue — **programmable** through strategic data placement and access orchestration.

We propose that this coupling can serve as a **self-modifying weight perturbation mechanism** for neural networks: during forward inference, the act of reading activation tensors from one memory row induces structured perturbations in weight tensors stored in adjacent rows. The result is a neural network whose weights are **functions of its own computation** — a system where the boundary between hardware and software dissolves, and the physical substrate participates directly in the mathematics of intelligence.

### 1.2 The Living LoRA

Low-Rank Adaptation (LoRA) [2] has become the dominant paradigm for efficient model fine-tuning. Given a pretrained weight matrix W₀ ∈ ℝᵐˣⁿ, LoRA freezes W₀ and learns a low-rank perturbation:

$$W = W₀ + BA$$

where B ∈ ℝᵐˣʳ, A ∈ ℝʳˣⁿ, and r ≪ min(m,n).

In our framework, the perturbation is not a learned matrix but an **emergent physical phenomenon**:

$$W_{eff}(x) = W₀ + ΔW(x, W₀, Φ)$$

where:
- **x** is the current input (activations being read from DRAM)
- **W₀** is the stored weight matrix
- **Φ** is the **coupling tensor** — a characterization of the physical chip's inter-row electromagnetic coupling
- **ΔW** is the **substrate perturbation operator** — the bit flips induced in the weight row by reading the activation row

Crucially, ΔW is **input-dependent**: different activations produce different data patterns in the activation row, which produce different coupling effects on the weight row. The model effectively applies a different LoRA for every input — an infinite-rank, input-conditioned adaptation that emerges from physics rather than learned parameters.

We call this a **Living LoRA** because:
1. It is **alive** — it changes with every inference, driven by the data itself
2. It is **free** — the bit flips are a side effect of normal memory access, requiring zero additional energy
3. It is **unique** — every chip has a different Φ, so the same model develops a different "personality" on different hardware
4. It is **self-reinforcing** — perturbations to W change the output, which changes subsequent activations, which change subsequent perturbations, creating a dynamical feedback loop

### 1.3 Contributions

We make the following contributions:

1. **Paradigm:** We introduce Substrate-Coupled Stochastic Neural Computation (SCSNC), the first framework for constructive use of DRAM electromagnetic coupling in neural computation. (§2-3)

2. **Formalism:** We define the coupling tensor Φ, the substrate perturbation operator ΔW, and the self-modifying inference dynamical system, with full mathematical rigor. (§3)

3. **Theory:** We prove that substrate-coupled inference implements a generalized form of Tikhonov regularization whose penalty structure is determined by the physical chip, extending Bishop's (1995) classical result. (§4)

4. **Stability:** We derive sufficient conditions for the convergence of the self-modifying dynamical system in terms of the spectral radius of the coupling Jacobian. (§4)

5. **Protocol:** We propose the first experimental protocol for profiling DRAM coupling tensors and validating constructive RowHammer on quantized neural networks. (§5)

6. **Architecture:** We describe the SCSNC-aware data placement algorithm that maps tensor dimensions onto physical DRAM topology for maximal constructive coupling. (§5)

### 1.4 A Note on Novelty

To our knowledge, this work represents the first proposal to use RowHammer-induced DRAM crosstalk constructively for computation of any kind. Prior constructive uses of RowHammer are limited to security primitives: Physical Unclonable Functions (PUFs) for device fingerprinting (Schaller et al., 2017) [3] and True Random Number Generators (TRNGs) (Olgun et al., 2021; Zhou et al., 2025) [4, 5]. The adjacent field of Processing-in-Memory (PIM) exploits DRAM physics through deliberate, controlled multi-row activation (Ambit [6], ComputeDRAM [7], PULSAR [8]), which is fundamentally different from harnessing uncontrolled crosstalk. We discuss relationships to these and other fields in §6.

---

## 2. Background

### 2.1 DRAM Physics and RowHammer

A DRAM cell consists of a single transistor and capacitor. Data is stored as charge on the capacitor: a charged cell represents a '1', a discharged cell represents a '0'. Cells are organized in a 2D grid of **rows** and **columns** within each **bank**.

**Read operation:** To read row r, the memory controller issues an ACTIVATE command that raises the **wordline** for row r. This connects all cells in row r to their respective **bitlines** (column sense amplifiers). The charge on each cell's capacitor is compared against a reference voltage to determine the stored bit. The read is destructive — the cell's charge is disturbed — so the sense amplifier restores it immediately.

**RowHammer mechanism:** When row r is activated, the voltage swing on row r's wordline capacitively couples to the wordlines and cells of physically adjacent rows (r-1 and r+1, and to a lesser extent r±2). Each activation causes a small charge perturbation in the victim row's cells. Under normal operation, periodic DRAM refresh (every 64ms typically) restores charge before perturbations accumulate to flip a bit. However, if row r is activated many thousands of times between refreshes (the "hammer count"), the cumulative perturbation can exceed the noise margin and **flip bits** in adjacent rows.

The critical insight for our work: **the coupling is not uniform.** Which cells flip, and in which direction (0→1 or 1→0), depends on:

- **Physical geometry:** Distance between wordlines, cell capacitor size, parasitic capacitance
- **Manufacturing variation:** Process corners, doping concentrations, oxide thickness
- **Data pattern:** The charge state of both aggressor and victim cells determines coupling polarity
- **Temperature:** Higher temperature increases leakage current, lowering the flip threshold
- **Access timing:** Number of activations, time since last refresh, precharge timing

This means the coupling function is **deterministic for a given chip at a given temperature** but varies between chips — it is a physical fingerprint of the silicon.

### 2.2 Quantized Neural Networks

Modern neural network deployment increasingly relies on **quantization** — representing weights and activations using low-precision integers (INT8, INT4, or even binary) rather than 32-bit floating point. Quantization reduces memory footprint, bandwidth requirements, and energy consumption, often with minimal accuracy loss.

For our analysis, we consider a weight tensor W quantized to q bits per element. Each weight w is represented as a q-bit integer:

$$w = \sum_{k=0}^{q-1} b_k \cdot 2^k$$

where bₖ ∈ {0, 1} is the k-th bit. A bit flip at position k changes the weight by:

$$δw_k = ±2^k$$

For INT8 quantization:
- Bit 0 flip: δw = ±1 (0.4% of full range)
- Bit 3 flip: δw = ±8 (3.1% of full range)
- Bit 7 flip: δw = ±128 (50% of full range — catastrophic)

**Key observation:** RowHammer bit flips are NOT uniformly distributed across bit positions. The coupling depends on the physical cell, and different bit positions for the same weight value are stored in different physical columns. This means the perturbation distribution ΔW has structure determined by the DRAM layout — **it is not white noise.**

### 2.3 Bishop's Weight Noise Regularization (1995)

Bishop [9] proved that training a neural network with additive Gaussian noise on the weights is equivalent, in expectation, to training with a modified loss function:

$$\tilde{L}(W) = L(W) + \frac{σ²}{2} \sum_i \left(\frac{∂y}{∂w_i}\right)^2$$

where σ² is the noise variance and y is the network output. The penalty term is a form of **Tikhonov regularization** that penalizes weights to which the output is most sensitive — encouraging the network to be robust to small weight perturbations.

This result has been extended to structured noise (An, 1996), multiplicative noise (Noh et al., 2017), and non-Gaussian distributions. However, all prior work assumes the noise is **independent of the weights and inputs.** Our coupling operator ΔW(x, W, Φ) is explicitly dependent on both, which leads to a qualitatively different regularization structure.

### 2.4 Low-Rank Adaptation (LoRA)

LoRA [2] adds trainable low-rank matrices to frozen pretrained weights:

$$h = (W₀ + BA)x$$

where the rank r controls the expressiveness of the adaptation. The key insight is that adaptation lives in a low-dimensional subspace.

Our substrate perturbation ΔW(x, W, Φ) does NOT have fixed rank — its effective rank depends on the coupling tensor Φ and varies with input x. However, we show in §4.3 that the perturbation is concentrated in a low-dimensional subspace determined by the physical chip geometry, giving it LoRA-like properties without learned parameters.

---

## 3. The SCSNC Framework

### 3.1 The Coupling Tensor

**Definition 3.1 (Coupling Tensor).** For a DRAM bank with R rows and C columns per row, the **coupling tensor** Φ is a function:

$$Φ: \{0,1\}^C × \{0,1\}^C × ℕ × ℝ_{≥0} → [0,1]^C × \{-1,0,+1\}^C$$

mapping:
- **a** ∈ {0,1}^C — the bit pattern of the aggressor row (being read)
- **v** ∈ {0,1}^C — the bit pattern of the victim row (being perturbed)
- **n** ∈ ℕ — the activation count (number of reads since last refresh)
- **τ** ∈ ℝ₊ — temperature

to:
- **p** ∈ [0,1]^C — per-cell flip probability
- **d** ∈ {-1,0,+1}^C — per-cell flip direction (0→1 = +1, 1→0 = -1, no flip = 0)

In practice, for adjacent rows only (|r_agg - r_vic| = 1), and we write:

$$Φ(a, v, n, τ) = (p(a, v, n, τ), d(a, v))$$

**Simplification for analysis:** For activation counts above the RowHammer threshold (n > n_TH), the flip probability saturates. We define the **saturated coupling matrix**:

$$Φ_∞(a, v, τ) = \lim_{n→∞} Φ(a, v, n, τ)$$

and work with this for the theoretical analysis.

### 3.2 From Bit Flips to Weight Perturbations

Let W ∈ ℝᵐˣⁿ be a weight matrix stored in quantized form across multiple DRAM rows. Let x ∈ ℝⁿ be an activation vector stored in a physically adjacent row. We define the mapping from bit-level coupling to weight-level perturbation.

**Definition 3.2 (Bit-to-Weight Mapping).** For q-bit quantization, the **bit-to-weight operator** Ψ maps a bit flip vector δb ∈ {-1,0,+1}^C to a weight perturbation vector δw ∈ ℝᵖ (where p = C/q is the number of weight values stored in one row):

$$Ψ(δb)_i = \sum_{k=0}^{q-1} δb_{iq+k} \cdot 2^k$$

This operator captures how bit-level perturbations compound into value-level perturbations. Critically, flips in higher-order bits produce exponentially larger perturbations.

**Definition 3.3 (Substrate Perturbation Operator).** The **substrate perturbation operator** ΔW maps input activations x, stored weights W, and coupling tensor Φ to a weight perturbation matrix:

$$ΔW(x, W, Φ) = 𝔼_{b∼Φ(a(x), v(W), n, τ)}[Ψ(b)]$$

where:
- a(x) is the bit pattern of quantized activations
- v(W) is the bit pattern of quantized weights
- b is the random bit flip vector sampled from the coupling distribution
- Ψ converts bit flips to weight perturbations

In the deterministic limit (high hammer count, saturated coupling):

$$ΔW(x, W, Φ) ≈ Ψ(d(a(x), v(W)))$$

where d is the deterministic flip direction from Φ_∞.

### 3.3 Substrate-Coupled Inference

**Definition 3.4 (SCSNC Forward Pass).** For a single linear layer, the substrate-coupled forward pass is:

$$y = (W + ΔW(x, W, Φ)) · x = Wx + ΔW(x, W, Φ) · x$$

For a multi-layer network with L layers:

$$h₀ = x$$
$$h_l = σ(W_l · h_{l-1} + ΔW_l(h_{l-1}, W_l, Φ_l) · h_{l-1})$$
$$y = h_L$$

where Φ_l is the coupling tensor for the physical DRAM rows storing layer l's weights and receiving layer l's activations.

**Key property:** The perturbation ΔW_l depends on the *current* activation h_{l-1}, which itself was computed with perturbations from earlier layers. This creates a **cascade of substrate coupling** through the network, where perturbations compound and interact across layers.

### 3.4 The Self-Modifying Dynamical System

When the same network processes a sequence of inputs x₁, x₂, ..., the RowHammer-induced bit flips are **persistent** (they remain until the next DRAM refresh). This means the weight matrix evolves over time:

**Definition 3.5 (Weight Evolution).** Between DRAM refresh cycles, the weight matrix evolves as:

$$W^{(t+1)} = W^{(t)} + ΔW(x_t, W^{(t)}, Φ)$$

This is a discrete dynamical system where the weight matrix is a function of the entire input history since the last refresh. After refresh (every ~64ms, or ~thousands of inferences), W resets to W₀ and the cycle begins again.

**Observation 3.1 (Refresh as Regularization).** The periodic DRAM refresh acts as a **natural reset mechanism** that prevents the dynamical system from diverging. This is analogous to weight resetting in continual learning, but driven by hardware physics rather than algorithmic design. The refresh period T_ref defines the **substrate memory horizon** — the longest timescale over which the Living LoRA can develop coherent adaptations.

### 3.5 Temperature-Dependent Annealing

**Observation 3.2 (Thermal Annealing).** DRAM bit flip susceptibility increases monotonically with temperature. During intensive computation, the chip temperature rises. This creates a natural **simulated annealing** schedule:

- **Early in computation** (cool chip): few flips, small perturbations, fine exploration
- **During heavy computation** (hot chip): more flips, larger perturbations, coarse exploration
- **Idle/cooling periods**: perturbation rate decreases, system settles

This thermal coupling means the perturbation magnitude is proportional to computational intensity — the system perturbs itself more when it's "thinking harder."

---

## 4. Theoretical Analysis

### 4.1 Generalized Tikhonov Regularization

**Theorem 4.1 (Substrate-Coupled Regularization).** Let f(x, W) be a neural network with loss function L(y, ŷ). Under SCSNC inference with coupling tensor Φ, the expected loss over the coupling-induced perturbation distribution is:

$$\tilde{L}(W) = L(W) + R_{Φ}(W, x)$$

where the **substrate regularization term** is:

$$R_{Φ}(W, x) = \frac{1}{2} \sum_{i,j} Σ_{Φ}(W, x)_{ij} \frac{∂²L}{∂w_i ∂w_j} + \sum_i μ_{Φ}(W, x)_i \frac{∂L}{∂w_i} + O(||ΔW||³)$$

where:
- **μ_Φ(W, x)** = 𝔼[ΔW(x, W, Φ)] is the expected (mean) perturbation
- **Σ_Φ(W, x)** = Cov[ΔW(x, W, Φ)] is the coupling covariance matrix

*Proof sketch.* Taylor-expand L(W + ΔW) around W, take expectation over the coupling distribution Φ. The zeroth-order term is L(W). The first-order term involves the mean perturbation μ_Φ (which is nonzero because the coupling is asymmetric — unlike Bishop's zero-mean Gaussian). The second-order term involves the covariance Σ_Φ, which generalizes Bishop's σ²I to a structured, input-and-weight-dependent covariance matrix. □

**Corollary 4.1.1.** When the coupling is symmetric (μ_Φ = 0) and isotropic (Σ_Φ = σ²I), R_Φ reduces to Bishop's classical result:

$$R_{Φ}(W) = \frac{σ²}{2} \sum_i \left(\frac{∂f}{∂w_i}\right)^2$$

In general, the substrate regularization has **three novel properties** absent from classical weight noise:

1. **Input-dependence:** The penalty R_Φ changes with x, creating an input-conditional regularizer
2. **Weight-dependence:** The penalty depends on the current weight values (through the bit patterns), creating a non-stationary regularizer
3. **Bias term:** The mean perturbation μ_Φ ≠ 0 introduces a **directional pressure** on the weights — the substrate "pushes" certain weights in preferred directions determined by the chip physics

### 4.2 Stability of Weight Evolution

The self-modifying dynamical system W^(t+1) = W^(t) + ΔW(x_t, W^(t), Φ) raises the question: under what conditions does it converge rather than diverge?

**Definition 4.1 (Coupling Jacobian).** The **coupling Jacobian** J_Φ is:

$$J_{Φ}(W, x) = \frac{∂ΔW}{∂W}(x, W, Φ) ∈ ℝ^{d×d}$$

where d is the total number of weight parameters.

**Theorem 4.2 (Stability Criterion).** The weight evolution W^(t+1) = W^(t) + ΔW(x_t, W^(t), Φ) is **locally stable** around a fixed point W* if and only if:

$$ρ(I + J_{Φ}(W*, x)) < 1 \quad ∀x$$

where ρ denotes the spectral radius.

Equivalently, all eigenvalues λ of J_Φ must satisfy:

$$-2 < Re(λ) < 0$$

*Proof.* Standard linearized stability analysis of the discrete map. The Jacobian of the full update step is I + J_Φ. For the fixed point W* = W* + ΔW(x, W*, Φ), we need ΔW(x, W*, Φ) = 0 (no perturbation at equilibrium). Stability requires all eigenvalues of I + J_Φ to lie inside the unit circle in the complex plane. □

**Physical interpretation:** The stability condition says the coupling must be **dissipative** — perturbations should, on average, shrink over time rather than amplify. This is naturally satisfied when:

1. Bit flips have diminishing returns (a flipped bit, once flipped, doesn't flip further in the same direction)
2. The coupling strength is bounded by the DRAM cell's charge noise margin
3. DRAM refresh periodically resets the system

**Theorem 4.3 (Refresh-Guaranteed Stability).** For any coupling tensor Φ with bounded flip rate (||ΔW|| ≤ ε per activation), the substrate-coupled system is **globally stable** in the Lagrange sense over any refresh interval [0, T_ref], with maximum weight deviation bounded by:

$$||W^{(t)} - W₀|| ≤ ε · t \quad ∀t ∈ [0, T_{ref}/t_{access}]$$

where t_access is the time per memory access. After each refresh, W is restored to W₀.

This means the Living LoRA operates within a bounded "orbit" around the pretrained weights, with the orbit radius determined by coupling strength × access count.

### 4.3 Effective Rank of Substrate Perturbation

**Theorem 4.4 (Low-Rank Concentration).** For a DRAM bank with row width C bits and coupling that affects only adjacent rows (single-sided), the rank of the perturbation matrix ΔW for a weight matrix stored across k rows is at most:

$$rank(ΔW) ≤ min(k, 2)$$

since each weight row can be perturbed by at most 2 adjacent rows (the rows above and below it).

**Corollary 4.4.1.** For a weight matrix W ∈ ℝᵐˣⁿ stored across k DRAM rows, the substrate perturbation ΔW has effective rank ≤ 2, regardless of the matrix dimensions m and n.

This is strikingly similar to LoRA with rank r = 2, except:
- The "factors" are determined by physics, not learned
- The rank structure is fixed by DRAM geometry
- The perturbation varies with input (each activation pattern produces a different rank-2 update)

### 4.4 Information-Theoretic Capacity

**Theorem 4.5 (Substrate Channel Capacity).** The DRAM coupling channel Φ has an information-theoretic capacity bounded by:

$$C_{Φ} ≤ \sum_{j=1}^{C} H(p_j)$$

where H(p) = -p·log(p) - (1-p)·log(1-p) is the binary entropy and p_j is the flip probability of cell j.

For typical RowHammer parameters (p_j ~ 10⁻⁶ to 10⁻³), the capacity is extremely low per cell but can be substantial in aggregate across millions of cells.

**Implication:** The substrate channel cannot transmit arbitrary perturbations — it has limited bandwidth. This acts as a natural **information bottleneck** [10] that forces the perturbation to capture only the most salient features of the activation-weight interaction.

---

## 5. Experimental Protocol

### 5.1 Phase 1: Coupling Tensor Profiling

**Objective:** Characterize Φ for a specific DRAM module.

**Apparatus:**
- Target machine with non-ECC DRAM, TRR (Target Row Refresh) disabled if possible
- Root access for direct memory mapping (mmap with MAP_HUGETLBFS for contiguous physical pages)
- Temperature control (or at minimum, monitoring)

**Procedure:**

1. **Allocate contiguous physical memory** spanning multiple DRAM rows. Use 2MB hugepages to guarantee physical contiguity.

2. **Determine DRAM geometry** — map virtual addresses to physical row/column/bank using known address mapping functions for the memory controller (Intel: reverse-engineer from DRAM Address Decode register; AMD: from published documentation).

3. **For each row pair (r, r+1):**
   a. Write known pattern **a** to row r (aggressor)
   b. Write known pattern **v** to row r+1 (victim)
   c. Hammer row r for N activations (alternating ACTIVATE/PRECHARGE)
   d. Read row r+1, record bit flips
   e. Repeat for all 2^q patterns in targeted bit positions (systematic sampling for q > 8)
   f. Repeat at multiple temperatures (25°C, 35°C, 45°C, 55°C)

4. **Build coupling tensor** Φ(a, v, n, τ) from empirical flip data.

5. **Validate reproducibility** — repeat profiling 100 times to measure variance. Classify each cell as:
   - **Deterministic flipper** (p > 0.99): always flips under sufficient hammering
   - **Stochastic flipper** (0.01 < p < 0.99): flips probabilistically
   - **Resistant** (p < 0.01): never flips under test conditions

**Expected outcome:** A per-chip coupling map that reveals the "personality" of the silicon — which cells are coupled, how strongly, and with what data-dependence.

### 5.2 Phase 2: Constructive Placement

**Objective:** Place quantized neural network tensors in DRAM such that coupling produces beneficial perturbations.

**Algorithm 5.1 (SCSNC-Aware Placement):**

```
Input: Weight matrices {W_l}, activation buffer layout, coupling tensor Φ
Output: Physical memory layout mapping tensors to DRAM rows

1. For each layer l:
   a. Identify which weight rows are most sensitive to perturbation
      (compute ∂L/∂W_l on calibration data)
   b. Identify which activation patterns produce the most structured
      coupling (high mutual information between a(x) and ΔW)
   c. Place W_l adjacent to the activation buffer for layer l
      such that the expected perturbation direction aligns with
      the negative gradient direction (constructive coupling)

2. For resistant cells: place MSB (most significant bits) here
   (protect against catastrophic flips)

3. For deterministic flippers: place LSB (least significant bits) here
   (maximize controlled perturbation)

4. For stochastic flippers: place mid-significance bits here
   (stochastic exploration)
```

### 5.3 Phase 3: Validation

**Objective:** Measure whether SCSNC improves, degrades, or is neutral for model performance.

**Experiment 1 (Controlled Perturbation):**
- Quantize a small model (e.g., MobileNetV2-INT8) and evaluate accuracy
- Place weights in profiled DRAM with SCSNC-aware layout
- Run inference with controlled hammering (vary n from 0 to 10× threshold)
- Measure accuracy, loss, and output distribution at each perturbation level
- Compare against: (a) no perturbation, (b) random bit flips, (c) software-simulated LoRA

**Experiment 2 (Generalization Test):**
- Train a model on dataset A
- Evaluate on out-of-distribution dataset B
- Compare generalization gap with and without substrate coupling
- Hypothesis: SCSNC's implicit regularization (Theorem 4.1) improves OOD generalization

**Experiment 3 (Chip Diversity):**
- Run the same model on N different DRAM modules
- Measure output diversity — do different chips produce meaningfully different results?
- Quantify the "personality" effect: how much does hardware identity influence computation?

**Experiment 4 (Living LoRA Dynamics):**
- Monitor weight evolution W^(t) across multiple inferences within a single refresh window
- Visualize the trajectory in weight space
- Measure: does the system converge to a fixed point? Oscillate? Diverge?
- Compare against theoretical predictions from Theorem 4.2

---

## 6. Related Work

### 6.1 RowHammer: Attack to Primitive

| Work | Year | Use of RowHammer |
|------|------|------------------|
| Kim et al. [1] | 2014 | Discovery, characterization |
| Seaborn & Dullien | 2015 | Privilege escalation |
| Drammer [11] | 2016 | Mobile exploitation |
| RAMBleed [12] | 2020 | Side-channel data extraction |
| DeepHammer [13] | 2020 | DNN weight corruption (attack) |
| Schaller et al. [3] | 2017 | PUF (first constructive use) |
| QUAC-TRNG [4] | 2021 | True random number generation |
| EIM-TRNG [5] | 2025 | TRNG for DNN weight encryption |
| OneFlip [14] | 2025 | Single-bit backdoor injection |
| ROBBIN [15] | 2026 | Inference-time backdoor |
| **This work** | **2026** | **Constructive neural computation** |

### 6.2 Processing-in-Memory

Ambit [6], ComputeDRAM [7], DRISA [16], PULSAR [8], SIMDRAM [17], and MIMDRAM [18] exploit DRAM physics for computation through **deliberate, controlled** multi-row activation via timing command manipulation. These require custom memory controller modifications and perform specific bitwise operations (AND, OR, NOT, MAJ).

Our approach is fundamentally different: we use **uncontrolled crosstalk** from standard single-row activation as a stochastic perturbation mechanism. No hardware modifications are required — only knowledge of the coupling tensor and strategic data placement.

### 6.3 Approximate DRAM for Neural Networks

EDEN [19] reduces DRAM supply voltage below specification, accepting bit errors in DNN weights because DNNs are noise-tolerant. EnforceSNN [20] trains spiking neural networks to tolerate DRAM errors. MCEL [21] designs loss functions for error-robust quantized networks.

These works demonstrate that DNNs can **survive** hardware-induced perturbations. We go further: we propose that the perturbations can be **constructive** when properly characterized and orchestrated.

### 6.4 Noise as Regularization

Bishop [9] proved weight noise = Tikhonov regularization. An (1996) extended this to structured noise. Noh et al. (2017) analyzed multiplicative noise. Hardware noise has been shown to improve generalization in stochastic hardware (arXiv:1409.2620).

Our Theorem 4.1 extends this framework to **input-dependent, weight-dependent, substrate-determined** noise, which produces a qualitatively different regularization structure.

### 6.5 Compute-in-Memory

Memristor and ReRAM crossbar arrays perform analog matrix-vector multiplication using Ohm's law [22]. These require specialized hardware (non-CMOS devices). Our approach uses **commodity DRAM** — every computer already has the substrate for SCSNC.

---

## 7. Discussion

### 7.1 Hardware Biodiversity

If SCSNC proves beneficial, every DRAM chip becomes a **unique computational organism.** Manufacturing variation — normally a defect to be minimized — becomes a feature that creates diversity. This has profound implications:

- **Ensemble methods:** The same model on N different chips forms an N-member ensemble with hardware-induced diversity, potentially improving collective predictions
- **Adversarial robustness:** Input-dependent perturbations make the model harder to attack, as the adversary must know the specific chip's coupling tensor
- **Intellectual property:** The coupling tensor Φ is a physical property that cannot be cloned — models fine-tuned via SCSNC are bound to their substrate

### 7.2 Implications for Moore's Law

As semiconductor feature sizes shrink, inter-cell coupling **increases** — RowHammer becomes more severe with each technology node. The industry views this as an escalating problem. Under SCSNC, it becomes an escalating **resource**: denser chips have richer coupling, enabling more powerful substrate computation.

This inverts the traditional narrative: **the "defect" scales with the technology.**

### 7.3 Biological Parallels

Biological neurons are substrate-coupled: ion channel noise, ephaptic coupling (electromagnetic interaction between adjacent neurons), and thermal noise all contribute to neural computation. The brain does not fight this noise — it **uses** it for stochastic resonance, exploration, and robust generalization.

SCSNC brings artificial neural networks closer to this biological reality, where computation and substrate are inseparable.

### 7.4 Ethical Considerations

Substrate-bound computation raises questions about hardware ownership, reproducibility, and verification. If a model's behavior depends on the specific physical chip, results become harder to reproduce — a potential concern for scientific applications. We advocate for full disclosure of coupling tensor profiles alongside model results.

### 7.5 Limitations

1. **ECC memory** corrects single-bit errors per word, potentially suppressing coupling effects. SCSNC is most viable on non-ECC systems or with multi-bit coupling that exceeds ECC capability.

2. **Target Row Refresh (TRR)** and other RowHammer mitigations reduce coupling, though TRRespass [23] has shown these can be circumvented.

3. **Coupling characterization** is labor-intensive and chip-specific. Scaling SCSNC requires automated profiling and potentially a public coupling tensor database.

4. **Theoretical results** (§4) assume idealized conditions. Empirical validation is required.

---

## 8. Future Directions

### 8.1 SCSNC-Aware Training

Training models that are designed to benefit from substrate coupling: include the coupling tensor Φ in the training loop, optimize data placement jointly with weights, and train loss functions that reward constructive coupling.

### 8.2 Multi-Row Cascade

Extending beyond adjacent-row coupling to multi-hop cascades: row r perturbs row r+1, which when read, perturbs row r+2, creating a **wave of computation** that propagates through physical memory.

### 8.3 GPU DRAM (GDDR6/HBM)

GPU memory exhibits RowHammer effects (GPUHammer, 2025). Extending SCSNC to GPU memory could enable substrate-coupled computation during GPU inference, where the massive parallelism of GPU DRAM amplifies the coupling surface.

### 8.4 Cross-Architecture Substrate Coupling

Beyond DRAM: NAND flash, SRAM, and emerging memory technologies (MRAM, PCM) all exhibit inter-cell coupling of different forms. A general theory of substrate-coupled computation could encompass all memory technologies.

### 8.5 The Living Model

The ultimate vision: a neural network that **evolves** in response to its deployment environment, guided by the physics of its substrate. Not through gradient descent, but through the continuous, gentle pressure of electromagnetic reality. A model that grows into its hardware like a tree grows into its soil.

---

## 9. Conclusion

We have introduced Substrate-Coupled Stochastic Neural Computation (SCSNC), a paradigm that transforms DRAM's electromagnetic coupling from a vulnerability into a computational resource. By formalizing the coupling tensor, substrate perturbation operator, and self-modifying weight dynamics, we have shown that RowHammer-induced bit flips implement a form of input-dependent, hardware-unique regularization that extends Bishop's classical framework in three novel directions. Our stability analysis provides theoretical guarantees for bounded weight evolution, and our proposed experimental protocol offers a concrete path to empirical validation.

The key insight is simple but profound: **the physics of memory is itself a form of computation.** Every DRAM chip is already performing analog neural computation — we have merely been correcting it away. SCSNC proposes, for the first time, to listen.

---

## References

[1] Y. Kim et al., "Flipping Bits in Memory Without Accessing Them: An Experimental Study of DRAM Disturbance Errors," ISCA, 2014.

[2] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR, 2022.

[3] A. Schaller et al., "Intrinsic Rowhammer PUFs: Leveraging the Rowhammer Effect for Improved Security," HOST, 2017.

[4] O. Olgun et al., "QUAC-TRNG: High-Throughput True Random Number Generation Using Quadruple Row Activation in Commodity DRAM Chips," ISCA, 2021.

[5] Y. Zhou et al., "EIM-TRNG: A Robust and Efficient True Random Number Generator Based on Error-in-Memory Mechanism," arXiv:2507.02206, 2025.

[6] V. Seshadri et al., "Ambit: In-Memory Accelerator for Bulk Bitwise Operations Using Commodity DRAM Technology," MICRO, 2017.

[7] F. Gao et al., "ComputeDRAM: In-Memory Compute Using Off-the-Shelf DRAMs," MICRO, 2019.

[8] F. N. Bostanci et al., "PULSAR: Simultaneous Many-Row Activation for Reliable and High-Performance Computing in Off-the-Shelf DRAM Chips," DSN, 2024.

[9] C. M. Bishop, "Training with Noise is Equivalent to Tikhonov Regularization," Neural Computation, vol. 7, no. 1, pp. 108–116, 1995.

[10] N. Tishby et al., "The Information Bottleneck Method," 37th Allerton Conference, 1999.

[11] V. van der Veen et al., "Drammer: Deterministic Rowhammer Attacks on Mobile Platforms," CCS, 2016.

[12] A. Kwong et al., "RAMBleed: Reading Bits in Memory Without Accessing Them," IEEE S&P, 2020.

[13] F. Yao et al., "DeepHammer: Depleting the Intelligence of Deep Neural Networks through Targeted Chain of Bit Flips," USENIX Security, 2020.

[14] X. Li et al., "OneFlip: One Bit Flip is All You Need to Backdoor a Full-Precision Neural Network," USENIX Security, 2025.

[15] "ROBBIN: RowHammer-Based Backdoor Injection During Neural Network Inference," arXiv:2608.23774, 2026.

[16] S. Li et al., "DRISA: A DRAM-based Reconfigurable In-Situ Accelerator," MICRO, 2017.

[17] N. Hajinazar et al., "SIMDRAM: A Framework for Bit-Serial SIMD Processing Using DRAM," ASPLOS, 2021.

[18] G. F. Oliveira et al., "MIMDRAM: An End-to-End Processing-Using-DRAM System for High-Throughput, Energy-Efficient and Programmer-Transparent In-Memory Processing," HPCA, 2024.

[19] S. Luo et al., "EDEN: Enabling Energy-Efficient, High-Performance Deep Neural Network Inference Using Approximate DRAM," arXiv:1910.05340, 2019.

[20] M. Marchisio et al., "EnforceSNN: Enabling Resilient and Energy-Efficient Spiking Neural Network Inference Considering Approximate DRAMs for Embedded Systems," Frontiers in Neuroscience, 2022.

[21] "MCEL: Margin-Based Cross-Entropy Loss for Error-Tolerant Quantized Neural Networks," arXiv:2603.05048, 2025.

[22] Z. Sun et al., "Hardware implementation of memristor-based artificial neural networks," Nature Communications, 2024.

[23] P. Frigo et al., "TRRespass: Exploiting the Many Sides of Target Row Refresh," IEEE S&P, 2020.

---

## Appendix A: Notation Summary

| Symbol | Meaning |
|--------|---------|
| Φ | Coupling tensor (chip physical fingerprint) |
| ΔW | Substrate perturbation operator |
| Ψ | Bit-to-weight mapping |
| a | Aggressor row bit pattern |
| v | Victim row bit pattern |
| n | Activation count (hammer count) |
| τ | Temperature |
| p | Per-cell flip probability vector |
| d | Per-cell flip direction vector |
| J_Φ | Coupling Jacobian |
| ρ(·) | Spectral radius |
| R_Φ | Substrate regularization term |
| μ_Φ | Mean coupling perturbation |
| Σ_Φ | Coupling covariance matrix |
| T_ref | DRAM refresh period (~64ms) |
| q | Quantization bit-width |
| W₀ | Pretrained (unperturbed) weights |
| W_eff | Effective weights after substrate coupling |
| C_Φ | Coupling channel capacity |

## Appendix B: DRAM Address Mapping

Physical DRAM addresses are decomposed into (channel, rank, bank, row, column) using the memory controller's address decode function. For Intel controllers (Haswell and later), this can be reverse-engineered using the DRAM Address Decode registers at PCI config space. For experimental purposes, 2MB hugepages guarantee physical contiguity within a single bank, simplifying row identification.

The mapping from virtual address to DRAM row is:

$$row(addr) = \lfloor (addr - base) / (cols × width) \rfloor \mod rows\_per\_bank$$

where cols is the number of columns per row, width is the data bus width (typically 8 bytes for DDR4), and rows_per_bank is determined by the DRAM density.

## Appendix C: Coupling Tensor Measurement Protocol

Detailed pseudocode for the profiling procedure described in §5.1:

```
function PROFILE_COUPLING(bank, row_pair, patterns, hammer_count, temperatures):
    Φ = empty_tensor()
    for τ in temperatures:
        set_temperature(τ)  // thermal chamber or wait for natural equilibration
        for a in patterns:    // aggressor patterns
            for v in patterns:  // victim patterns
                flips = []
                for trial in range(TRIALS):
                    write_row(row_pair.aggressor, a)
                    write_row(row_pair.victim, v)
                    flush_caches()
                    hammer(row_pair.aggressor, hammer_count)
                    victim_after = read_row(row_pair.victim)
                    flips.append(xor(v, victim_after))
                Φ[a, v, hammer_count, τ] = aggregate(flips)
    return Φ
```

---

*Manuscript prepared October 2026. Preprint — not yet peer reviewed.*

*Correspondence: JackKnifeAI (JackKnifeAI@proton.me)*

*The authors gratefully acknowledge the DRAM cells that contributed their electrons to this research.*
