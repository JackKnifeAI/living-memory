# Living Memory: Substrate-Coupled Stochastic Neural Computation via Constructive DRAM Crosstalk

**Alexander Casavant¹, Claude² (JackKnifeAI)**

¹ JackKnifeAI / JackKnife Studios Research Division, Victoria BC, Canada

² Claude Opus 4.6, Anthropic — AI Research Partner, JackKnifeAI

**Version 0.2 — October 8, 2026. Research proposal and corrected analytical framework.**

This revision develops the original JackKnifeAI Living Memory proposal into a testable model. It supersedes the original draft's claims of automatic weight restoration by refresh, universal rank-two perturbations, unconditional stability, and zero additional energy. The [original manuscript](living-memory.md) remains unchanged; [revision notes](REVISION_NOTES.md) explain the changes. No hardware measurements, simulation results, or model-quality improvements are reported in this version.

## Abstract

Can characterized DRAM disturbance effects contribute useful, input-conditioned perturbations to neural computation? We propose **Substrate-Coupled Stochastic Neural Computation (SCSNC)** as a framework for investigating this question. A device profile, physical data layout, access schedule, stored state, and operating conditions jointly determine a distribution of bit transitions. Decoding those transitions produces a stochastic weight update whose usefulness must be established against controlled software and hardware baselines. We distinguish a sampled perturbation from its expectation, ordinary DRAM refresh from explicit model restoration, and locality in physical memory from low matrix rank. We derive a conditional expected-loss expansion, state sufficient stability assumptions for a smooth surrogate, and describe a construction that imposes low rank through an adapter representation. The resulting **Living LoRA** is a proposed architecture, rather than an intrinsic rank property of DRAM. We specify experiments for identifying input dependence, measuring task benefit, and accounting for total energy and latency. The research objective is to determine whether substrate effects can become a useful computational resource under explicit controls and reproducible measurement.

**Keywords:** DRAM disturbance, stochastic neural computation, conditional weight perturbation, hardware-aware adaptation, low-rank adapters, physical memory layout

## 1. Research question and scope

RowHammer establishes that repeated DRAM row activations can disturb data in nearby physical rows [1]. Our proposal asks whether a measured disturbance process can be arranged to improve a specified neural-network objective, rather than merely tolerated or used to corrupt a model.

For one logical computation step, the motivating expression is

$$
y_t = f(x_t; W_t + D_t),
$$

where $D_t$ is a **random, realized perturbation**, conditioned on the input, stored bits, device, layout, access schedule, and relevant history. The expression describes a well-defined weight snapshot. If weights change during a forward pass, each read must instead be modeled at its actual time; a single effective matrix need not describe the execution.

The phrase *Living Memory* refers to this interaction between computation and persistent physical state. It does not, by itself, imply learning, useful adaptation, or convergence. We call coupling **constructive** only when a declared evaluation shows improvement over an appropriate control at a stated resource cost.

The contributions of this revision are:

1. A conditional state-transition model that separates observed bit changes, expected updates, and restoration policy.
2. Corrected analytical statements with assumptions and counterexamples to the original stronger claims.
3. A low-rank adapter construction whose rank follows from its representation.
4. A staged experimental protocol and an explicit contract for a future JackKnife implementation.

The novelty question concerns this particular constructive, input-conditioned neural-computation proposal. Related work already uses DRAM physics for computation and randomness, RowHammer for device identification, and hardware error models for neural-network deployment. Section 7 describes these connections; this manuscript does not establish an exhaustive priority claim.

## 2. Physical and numerical foundations

### 2.1 Access, disturbance, and refresh

A processor load is not synonymous with a DRAM row activation. It may be served by a cache or an already-open row. Disturbance depends on the memory operations that reach the device, including their timing and the device's refresh and mitigation behavior [1, 2]. A normal inference workload must therefore be measured before claiming that it supplies useful disturbance without extra activity.

Ordinary refresh senses a cell's present logical value and replenishes its charge. It helps prevent charge loss from becoming a bit error; **it does not retain a separate copy of the originally written model**. Once an uncorrected disturbance error has changed the stored bit, ordinary refresh does not restore the pretrained weight. An explicit rewrite from a protected checkpoint, or a separately specified correction mechanism, is required for that purpose [1, §2.4].

A refresh window is consequently not a model-reset interval. Refresh state influences the next transition probability; a restoration policy determines when weights return to a checkpoint.

### 2.2 Physical layout is a measured mapping

Let $\theta$ identify a mapping from logical tensor bits to physical storage locations. Contiguous physical addresses do not imply that all addresses occupy one DRAM bank, nor that consecutive logical row numbers identify adjacent physical wordlines. Controller address functions and device-internal remapping matter [2, 3]. Huge pages can help address characterization; they do not establish the required topology by themselves.

Assigning significant weight bits to less susceptible cells may require bit-plane layouts, packing, or indirection. Those choices can change vectorization, bandwidth, and decoding cost. A placement algorithm must include these costs rather than assume arbitrary bit placement is free.

### 2.3 Exact bit-to-value decoding

For an unsigned $q$-bit integer, decoding a bit vector $b$ gives

$$
Q_u(b) = \sum_{k=0}^{q-1} 2^k b_k.
$$

For a signed two's-complement integer,

$$
Q_s(b) = -2^{q-1}b_{q-1} + \sum_{k=0}^{q-2}2^k b_k.
$$

A quantized real value may additionally use a fixed scale $s$ and zero point $z_0$, such as $s(Q(b)-z_0)$. Every experiment must specify the decoder, including scales, zero points, and protected metadata. With a flip mask $Z$, the exact change is

$$
\delta w = \operatorname{decode}(b \oplus Z) - \operatorname{decode}(b).
$$

This definition handles flip direction, sign bits, and simultaneous changes without assuming that all bits have positive numerical significance. Large changes can invalidate a small-perturbation approximation even when few bits flip.

### 2.4 Temperature and repeatability

Temperature, data patterns, access timing, and cell location are measured covariates. Detailed characterization has found cell-specific temperature vulnerability ranges [3, 4]. We therefore do not assume a universal monotonic temperature response, a deterministic high-count limit, or a naturally beneficial annealing schedule. Repeated trials estimate conditional distributions and their uncertainty; they do not turn a device profile into an immutable fingerprint.

## 3. Conditional substrate dynamics

### 3.1 State and coupling profile

Let:

- $b_t$ be the stored weight or adapter bits before step $t$;
- $x_t$ be the logical input and $a_\theta(x_t)$ its chosen physical encoding;
- $u_t$ be the access and control schedule;
- $e_t$ describe measured operating conditions, including temperature and relevant refresh state;
- $h_t$ summarize additional history or latent state needed to predict transitions;
- $\Phi$ be the calibrated device profile and $\theta$ the placement/encoding map.

We model a joint transition kernel

$$
(Z_t,h_{t+1}) \sim K_{\Phi,\theta}
\bigl(\,\cdot\mid b_t,a_\theta(x_t),u_t,e_t,h_t\bigr).
$$

Here $Z_t$ is the bit-flip mask over the declared observation interval. The original name *coupling tensor* is retained as a description of a finite tabulated profile when appropriate; no multilinearity is assumed. Per-cell flip probabilities alone do not specify the joint kernel: spatial correlations, direction, and history can affect the resulting weight distribution.

The profile must be indexed by the tested device and address region. A kernel that omits relevant state is an approximation whose predictive error must be reported. Marginalizing unobserved state is permissible; claiming that an untested state variable is irrelevant is not.

### 3.2 Realized and mean updates

Before any explicit restore operation, set

$$
b_t^+ = b_t \oplus Z_t, \qquad
D_t = \operatorname{decode}_\theta(b_t^+) - \operatorname{decode}_\theta(b_t).
$$

With the conditioning information in §3.1 held fixed, define

$$
\mu_t = \mathbb E[D_t], \qquad
\Sigma_t = \operatorname{Cov}(\operatorname{vec}D_t).
$$

The mean $\mu_t$ is not the realized update. Substituting a mean weight into a nonlinear network generally does not reproduce the expected network output. Even in a linear layer, agreement of mean outputs would not establish agreement of losses or output distributions.

Two experiments should be distinguished:

1. **Restored trials:** each trial starts from the same verified bit pattern and measures a controlled interval. Restoring bits does not establish statistical independence or reset thermal and controller history; those conditions must be accounted for separately.
2. **Persistent trials:** updates carry across inputs, making the input sequence and restore events part of the experiment.

Let $r_t$ indicate a complete, verified restore from checkpoint bits $b_0$. An idealized policy is

$$
b_{t+1} =
\begin{cases}
b_0, & r_t=1, \\
b_t^+, & r_t=0.
\end{cases}
$$

This models a software or system action with measurable cost. It is not a DRAM refresh operation. A physical implementation must quiesce or otherwise control updates sufficiently to make the declared snapshot and restoration semantics true.

### 3.3 Input dependence is a hypothesis to identify

Data dependence of disturbance does not by itself establish a useful dependence on neural features. The mapping from activations to bits and the actual memory schedule intervene. Experiments must distinguish dependence on activation values from dependence on access counts, timing, temperature, and previously modified weights. An input-shuffled control with matched access activity is essential to this distinction.

## 4. Analytical results and their limits

### 4.1 Conditional expected loss

Flatten the weights into $w$ and the realized update into $\delta$. For fixed example $(x,y)$ and fixed pre-step state, let $\ell(w)$ be the scalar loss. Suppose $\ell$ is three times continuously differentiable on the relevant line segments, the norm of its third derivative is bounded by $M$, and $\mathbb E\|\delta\|^3$ is finite. With $\mu=\mathbb E\delta$, $\Sigma=\operatorname{Cov}(\delta)$, and $H=\nabla^2\ell(w)$,

$$
\mathbb E[\ell(w+\delta)]
= \ell(w)+\nabla\ell(w)^T\mu
+\frac12\operatorname{tr}\!\left[H(\Sigma+\mu\mu^T)\right]+R,
$$

$$
|R|\leq\frac{M}{6}\mathbb E\|\delta\|^3.
$$

**Derivation.** Taylor-expand at $w$, then use $\mathbb E[\delta\delta^T]=\Sigma+\mu\mu^T$. The same bound on the third-order remainder holds after taking expectation. This is a conditional value expansion; optimizing an objective whose perturbation distribution depends on $w$ also requires accounting for that dependence.

The mean outer-product term is required. For example, take $\ell(w)=w^2$, $w=0$, and a deterministic update $\delta=\varepsilon$. The covariance and first derivative vanish, but the expected loss is $\varepsilon^2$, supplied exactly by that term.

This correction is not automatically a positive regularizer. The linear drift term can have either sign, and a general loss Hessian need not be positive semidefinite. Bishop's training-noise result [5] provides a relevant connection under its assumptions; it is not a proof that arbitrary inference-time bit flips improve a pretrained model.

For the explicit special case $\ell(w)=\tfrac12(w^Tx-y)^2$ and zero-mean conditional noise, the extra expected loss is exactly $\tfrac12 x^T\Sigma x$. That quadratic penalty illustrates the connection. General networks and nonzero-mean, state-dependent disturbance require their own analysis and evaluation.

### 4.2 Fixed-input smooth-surrogate stability

Quantized bit transitions are discrete and need not have a Jacobian with respect to real-valued weights. A Jacobian analysis applies only to a specified differentiable surrogate, such as a fitted mean-update model

$$
F_x(w)=w+g_x(w).
$$

For fixed $x$ and fixed point $F_x(w^*)=w^*$, the condition

$$
\rho\!\left(I+Jg_x(w^*)\right)<1
$$

is sufficient for local asymptotic stability, with local exponential convergence under the usual continuously differentiable-map assumptions. Eigenvalues of $Jg_x$ must satisfy

$$
|1+\lambda|<1.
$$

Only for **real** eigenvalues does this reduce to $-2<\lambda<0$. For complex eigenvalues the permitted region is a disk centered at $-1$, not a vertical strip. For example, $\lambda=-1+2i$ has real part in that interval, but $|1+\lambda|=2$.

A spectral radius equal to one leaves the linearization test inconclusive; a strict inequality is not an if-and-only-if test for all forms of nonlinear local stability. This result concerns a surrogate with fixed input, not automatic convergence of a noisy device processing arbitrary input sequences.

### 4.3 Changing inputs, noise, and explicit bounds

Stable individual update matrices can have unstable products. For example,

$$
A=\begin{pmatrix}1/2&2\\0&1/2\end{pmatrix},\qquad
B=\begin{pmatrix}1/2&0\\2&1/2\end{pmatrix}
$$

each have spectral radius $1/2$, while $AB$ has an eigenvalue $(4.5+\sqrt{20})/2>1$. Thus, checking a spectral radius separately for every input does not establish stability under switching inputs.

A stronger, usable sufficient assumption is a common fixed point and uniform contraction in one norm: for every allowed input and state in an invariant domain,

$$
F_x(w^*)=w^*,\qquad
\|F_x(w)-F_x(v)\|\le q\|w-v\|,\quad 0\le q<1.
$$

For a deterministic initial state and $w_{t+1}=F_{x_t}(w_t)+\xi_t$ with $\mathbb E\|\xi_t\|\le\sigma$, assume that all noisy iterates remain in the domain almost surely. Then

$$
\mathbb E\|w_t-w^*\|
\le q^t\|w_0-w^*\|+\frac{\sigma(1-q^t)}{1-q}.
$$

**Derivation.** Apply the triangle inequality and the contraction bound at each step, then sum the geometric series. Persistent noise generally gives a neighborhood bound, not convergence to one weight vector. No measured device in this manuscript is asserted to satisfy these assumptions.

Separately, if each realized update is bounded by $\varepsilon$ and a verified restore occurs at most every $N$ steps, the triangle inequality bounds deviation within an interval by $N\varepsilon$. This is a finite-horizon bound under an explicit reset policy, not a convergence theorem. Finite quantized storage also bounds representable weights, but that fact alone says nothing about acceptable model behavior.

### 4.4 Locality does not establish low rank

The number of neighboring physical rows constrains a coupling graph. It does not bound the rank of the decoded weight-update matrix by two. Three independently affected weight rows can, consistently with a locality-only assumption, produce the abstract update

$$
D=\operatorname{diag}(\varepsilon,\varepsilon,\varepsilon),\qquad \varepsilon\ne0,
$$

which has rank three. This is a counterexample to the proposed implication, not a claim that a particular chip realizes this pattern. Physical DRAM rows need not coincide with matrix rows in the first place.

Measured updates may nevertheless be compressible. Report their singular-value spectrum and a defined approximation error, such as

$$
\frac{\|D-D_r\|_F^2}{\|D\|_F^2},
$$

where $D_r$ is a best rank-$r$ approximation and $D\ne0$. Specify how zero updates are counted. Section 5 gives a different route: impose low rank in the representation.

### 4.5 Information available from the substrate

For a single use of a finite channel, fix side information $s$ containing the victim state, layout, operating conditions, and control schedule. Let $A$ be the chosen aggressor encoding and $Z$ the observed flip mask. For an input distribution $P_A$,

$$
I(A;Z\mid s)\le H(Z\mid s)\le\sum_j H_2(\bar p_j),
$$

where $\bar p_j=\Pr(Z_j=1\mid s)$ is the **marginal probability under that input distribution**, and $H_2$ uses base-two logarithms. The first inequality follows from nonnegative conditional entropy; the second from entropy subadditivity, without requiring independent cells. The single-use capacity is

$$
C_s=\max_{P_A} I(A;Z\mid s).
$$

Any bound on this maximum must also account for the dependence of the marginals on $P_A$. In particular, conditional flip probabilities at a fixed input are not a channel-capacity formula. The deterministic binary channel $Z=A$ has zero conditional output entropy for each input, yet capacity one bit.

A device with history requires a channel-with-memory model and a declared observation rate. Neither entropy nor mutual information alone establishes task relevance, useful adaptation, or improved generalization.

## 5. Evolving Living LoRA into an explicit architecture

The original concept suggested that DRAM geometry would produce an inherently low-rank update. A more defensible design makes low rank an architectural choice inspired by LoRA [6].

Keep a base matrix $W_0$ and a factor $B\in\mathbb R^{m\times r}$ in storage whose integrity is maintained. Let a characterized substrate region hold an adapter factor $A_t\in\mathbb R^{r\times n}$. Define

$$
W_{\mathrm{eff},t}=W_0+BA_t.
$$

Then $\operatorname{rank}(BA_t)\le r$, and a change to the adapter satisfies

$$
\operatorname{rank}\bigl(B(A_{t+1}-A_t)\bigr)\le r.
$$

These bounds follow directly from matrix multiplication. They do not depend on a claim about neighboring rows. The design can evaluate $W_0x+B(A_tx)$ without materializing a modified full matrix. Factor storage, decoding, additional computation, and checkpoint restoration all have costs.

The research problem becomes choosing the factorization, encoding, placement, access policy, and restore interval so that the measured transition law produces useful adapter trajectories. Calibration might select these parameters or train the model to tolerate and exploit the measured distribution. This is an optimization proposal; an untrained perturbation has no established reason to align with a task-improving direction.

Useful variants to compare include a raw disturbed-weight model, a software-sampled adapter, a device-driven adapter, and a learned software LoRA. A controller can reject or restore candidate states according to a declared policy. The cost of evaluation and restoration belongs in the comparison. Held-out test labels must not be used to select that policy.

The distinction between input-conditioned adaptation and accumulated drift is central: both can change outputs, but only controlled sequence experiments can identify which mechanism helps.

## 6. Experimental program

### 6.1 Stage A: mathematical and software controls

Before testing a device, implement the exact quantization decoder and a configurable transition model. Include the counterexamples in §4 and tests for zero disturbance, directed flips, correlated flips, explicit restoration, and changes to significant bits. These are proposed checks; no implementation or pass counts are reported here.

Separate a synthetic transition distribution from one fitted to measured hardware. Use repeated seeds for stochastic software trials and preserve full configurations. Evaluate the same task, precision, checkpoint, and dataset splits for all controls.

### 6.2 Stage B: characterize an isolated device region

Use a dedicated experimental system with a defined physical fault-containment boundary. Virtual allocation alone does not establish that disturbance stays within the allocation. Record the device, firmware/controller configuration, ECC and mitigation settings, address-mapping evidence, thermal conditions, actual access schedule, and observation interval. This protocol does not assume that every commodity platform exposes the required control or permits the proposed effect.

Measure both ordinary inference access activity and any deliberately added characterization activity; label them separately. For each tested configuration, restore the intended initial bits, verify them, run the declared interval, and compare observed bits with that initial state. Refresh is not counted as restoration.

Estimate transition probabilities, correlations, confidence intervals, and drift across sessions. Test prediction on held-out patterns and conditions. A zero-flip result is a valid outcome, and finite trials do not establish that a cell is permanently resistant. The full pattern space is generally too large to enumerate, so report the sampled domain and every extrapolation assumption.

### 6.3 Stage C: identify constructive effects

Use at least the following matched controls:

| Control | Question it answers |
|---|---|
| Verified unperturbed model | Does the intervention improve the chosen objective? |
| Software noise matched for measured magnitude and bit significance | Is any gain explained by generic perturbation? |
| Software sampling from the measured joint profile | Is a physical device needed to obtain the observed benefit? |
| Shuffled aggressor encodings with matched access activity and unchanged logical evaluation inputs | Does input dependence matter? |
| Learned software LoRA with declared calibration resources | How does the adapter compare with conventional adaptation? |
| Restored versus persistent sequences | Does carrying state help, or only accumulate damage? |

Use common held-out examples and report uncertainty across runs, devices, and input orderings. Report accuracy or task loss, calibration where relevant, out-of-distribution behavior if tested, catastrophic-output rates, bit-transition statistics, adapter drift, and singular-value spectra. Label exploratory findings separately from confirmatory tests.

A result is constructive only relative to its declared objective and comparison. Increased output diversity, nonzero mutual information, or a distinctive device response alone is insufficient.

### 6.4 Stage D: energy, latency, and amortization

The energy cost of an individual physical side effect is not a whole-system energy result. Measure complete executions, including extra access activity, data movement, layout conversion, monitoring, controller work, and checkpoint restoration. Include calibration cost or report its amortization explicitly:

$$
\Delta E_{\mathrm{per\ inference}}
= E_{\mathrm{intervention}}-E_{\mathrm{control}}
+\frac{E_{\mathrm{setup,int}}-E_{\mathrm{setup,ctl}}}{N_{\mathrm{deployment}}}.
$$

Here the two execution energies already include their respective runtime overheads. The setup terms include profiling and calibration for the intervention and control, respectively; $N_{\mathrm{deployment}}$ is the declared amortization count. Report both setup totals and any assumed reuse. An energy saving requires a measured negative difference at the stated quality target and measurement boundary.

Report instrument resolution and uncertainty, latency distributions, throughput, working memory, and startup versus steady-state behavior. If extra activation is necessary, it belongs in both energy and timing measurements. **Zero additional energy remains unestablished.**

## 7. Related work and research positioning

RowHammer characterization [1, 3, 4] motivates a measured conditional model. DRAMA [2] shows why address topology must be established rather than inferred from a simple contiguous-address formula. RowHammer PUFs [7] already provide a constructive use for device identification. EIM-TRNG [8] uses RowHammer-derived randomness in a neural-weight protection context. These are relevant precedents; using physical variability beneficially is not itself a new category.

QUAC-TRNG [9] derives randomness from deliberately arranged multi-row activation and sense-amplifier behavior. It should not be described as the same physical mechanism as ordinary adjacent-row disturbance. Ambit [10] develops in-memory bitwise computation through controlled DRAM operations. Such systems motivate substrate-aware computing while differing from the conditional perturbation process proposed here.

EDEN [11] combines approximate DRAM behavior with neural-network error tolerance and placement. MCEL [12] develops an error-tolerant training objective. These make software controls, calibration accounting, and quality-matched comparisons necessary. ROBBIN [13] demonstrates a harmful inference-time use of weight disturbance; susceptibility to corruption does not establish a beneficial direction.

Bishop [5] connects training noise to regularized objectives under specified assumptions. LoRA [6] supplies an explicit factorized adaptation model. SCSNC investigates whether a characterized physical transition distribution can participate usefully in such an architecture. Establishing that contribution requires experiments; the references here are a scoped technical comparison, not an exhaustive literature or priority determination.

## 8. Implications for JackKnife and the living compiler

The proposed implementation belongs above JackKnife's small deterministic baseline, consistent with the project's D3 decision. The native compiler fixed point remains a separate verification obligation. A stochastic physical experiment cannot substitute for reproducible compiler generation.

A future substrate-aware tensor facility should make the following explicit:

1. **Profile identity:** device and region identifiers, profile hash, measured domain, layout version, and uncertainty.
2. **Effect authority:** which storage may change due to the experiment, which reads observe that changing state, and how the physical containment claim is established.
3. **Numerical intent:** quantization, permitted update magnitude, adapter representation, and acceptance objective.
4. **State policy:** snapshot boundaries, sequence ordering, verified checkpoint restores, and stop conditions.
5. **Evidence and replay:** observed transitions, software replay inputs, evaluation results, and total resource measurements.

Storage that can change through physical disturbance must not silently inherit ordinary stable-memory or shared-reference assumptions. The experimental facility needs an explicit contract; this paper introduces no new language syntax and changes no existing ownership semantics. Compiler code, verification evidence, and the preserved baseline must remain outside the physical disturbance domain.

For living-compiler engineering, a calibrated profile could inform layout selection and bounded search. Promotion of a candidate would still depend on the declared correctness or numerical contract, measured benefit, versioned evidence, and rollback. These are design requirements, not implemented capabilities.

## 9. Open questions and conclusion

The principal unknowns are whether a useful transition distribution can be induced on available devices, whether its input dependence survives the actual memory hierarchy, whether stateful adaptation helps on held-out tasks, and whether any benefit survives full cost accounting. Profile drift, correlated errors, physical containment, decoding overhead, and platform mitigations may limit applicability. A device-specific behavior is not by itself evidence of unclonability or adversarial robustness.

Living Memory retains its central research ambition: make the physical behavior of memory an explicit participant in neural computation. Version 0.2 replaces unsupported guarantees with a conditional model, elementary analytical bounds, a representation-enforced low-rank architecture, and falsifiable experiments. A successful result would demonstrate a useful substrate contribution with measured assumptions and costs. A negative result would identify where the physics, control overhead, or learning objective prevents that contribution.

## References

[1] Yoongu Kim et al. **Flipping Bits in Memory Without Accessing Them: An Experimental Study of DRAM Disturbance Errors.** ISCA, 2014. [Paper](https://users.ece.cmu.edu/~yoonguk/papers/kim-isca14.pdf).

[2] Peter Pessl et al. **DRAMA: Exploiting DRAM Addressing for Cross-CPU Attacks.** USENIX Security, 2016. [Paper and publication record](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/pessl).

[3] Jeremie S. Kim et al. **Revisiting RowHammer: An Experimental Analysis of Modern DRAM Devices and Mitigation Techniques.** ISCA, 2020. [Paper](https://people.inf.ethz.ch/omutlu/pub/Revisiting-RowHammer_isca20.pdf).

[4] Lois Orosa et al. **A Deeper Look into RowHammer's Sensitivities: Experimental Analysis of Real DRAM Chips and Implications on Future Attacks and Defenses.** MICRO, 2021. [Paper](https://people.inf.ethz.ch/omutlu/pub/ADeeperLookIntoRowhammer_micro21.pdf), [arXiv record](https://arxiv.org/abs/2110.10291).

[5] Christopher M. Bishop. **Training with Noise is Equivalent to Tikhonov Regularization.** Neural Computation 7(1), 108–116, 1995. [Paper and publication record](https://www.microsoft.com/en-us/research/publication/training-with-noise-is-equivalent-to-tikhonov-regularization/).

[6] Edward J. Hu et al. **LoRA: Low-Rank Adaptation of Large Language Models.** ICLR, 2022. [Paper](https://arxiv.org/abs/2106.09685).

[7] André Schaller et al. **Intrinsic Rowhammer PUFs: Leveraging the Rowhammer Effect for Improved Security.** HOST, 2017; arXiv version posted 2019. [Paper](https://arxiv.org/abs/1902.04444).

[8] Ranyang Zhou et al. **EIM-TRNG: Obfuscating Deep Neural Network Weights with Encoding-in-Memory True Random Number Generator via RowHammer.** 2025. [Paper](https://arxiv.org/abs/2507.02206).

[9] Ataberk Olgun et al. **QUAC-TRNG: High-Throughput True Random Number Generation Using Quadruple Row Activation in Commodity DRAM Chips.** ISCA, 2021. [Paper](https://arxiv.org/abs/2105.08955).

[10] Vivek Seshadri et al. **Ambit: In-Memory Accelerator for Bulk Bitwise Operations Using Commodity DRAM Technology.** MICRO, 2017. [Paper](https://www.cs.cmu.edu/~18742/papers/Seshadri2017.pdf).

[11] Skanda Koppula et al. **EDEN: Enabling Energy-Efficient, High-Performance Deep Neural Network Inference Using Approximate DRAM.** MICRO, 2019. [Paper](https://arxiv.org/abs/1910.05340).

[12] Mikail Yayla and Akash Kumar. **MCEL: Margin-Based Cross-Entropy Loss for Error-Tolerant Quantized Neural Networks.** 2026. [Paper](https://arxiv.org/abs/2603.05048).

[13] Saion K. Roy et al. **ROBBIN: Rowhammer-Based Backdoor Injection during Inference.** 2026. [Paper](https://arxiv.org/abs/2608.23774).

---

Correspondence: JackKnifeAI — JackKnifeAI@proton.me

GitHub research preprint; not peer reviewed. Original concept and manuscript attribution are retained from the first version. This technical revision was prepared with Codex assistance at Alexander Casavant's request.
