# DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation

**Makoto Shing**¹, **Masanori Koyama**², **Takuya Akiba**¹

¹Sakana AI, ²The University of Tokyo

`{mkshing,takiba}@sakana.ai`, `masanori.koyama@weblab.t.u-tokyo.ac.jp`

---

## Abstract

End-to-end backpropagation requires storing activations throughout all layers, creating memory bottlenecks that limit model scalability. Existing block-wise training methods offer means to alleviate this problem, but they rely on ad-hoc local objectives and remain largely unexplored beyond classification tasks. We propose _DiffusionBlocks_, a principled framework for transforming transformer-based networks into genuinely independent trainable blocks that maintain competitive performance with end-to-end training. Our key insight leverages the fact that residual connections naturally correspond to updates in a dynamical system. With minimal modifications to this system, we can convert the updates to those of a denoising process, where each block can be learned independently by leveraging the score matching objective. This independence enables training with gradients for only one block at a time, thereby reducing memory requirements in proportion to the number of blocks. Our experiments on a range of transformer architectures (vision, diffusion, autoregressive, recurrent-depth, and masked diffusion) demonstrate that DiffusionBlocks training matches the performance of end-to-end training while enabling scalable block-wise training on practical tasks beyond small-scale classification. DiffusionBlocks provides a theoretically grounded approach that successfully scales to modern generative tasks across diverse architectures.

Code is available at: <https://github.com/SakanaAI/DiffusionBlocks>

---

## 1. Introduction

### The memory bottleneck in neural network training

Modern AI led by generative models has become integral to everyday life. These models rely on _end-to-end backpropagation_, which requires storing intermediate activations across network layers during training. This fundamental requirement causes memory consumption to grow linearly with network depth, creating computational bottlenecks that limit both research flexibility and practical deployment.

### Block-wise training: promises and limitations

_Block-wise training_ methods[^1] partition networks into smaller components that can be trained independently, promising dramatic memory savings. Despite this potential, existing approaches consistently underperform end-to-end training. The core challenge is twofold: (1) **lack of theoretical grounding**: existing methods rely on ad-hoc local objectives without principled coordination between blocks, (2) **limited applicability**, where they require paradigm-specific designs, task-specific objectives that do not naturally extend beyond classification. Their results are typically demonstrated only on custom architectures without providing systematic procedures to be applied to modern architectures such as Transformers, leaving their applicability to modern generative AI largely unexplored. Without a systematic framework grounded in theory, block-wise training remains an unfulfilled promise.

[^1]: We use _block-wise training_ to encompass all approaches that partition networks into independently trainable components. This includes _layer-wise training_ as the special case where each block contains one layer.

### Diffusion models: a mathematical foundation for decomposition

Score-based diffusion models model the data distribution through a continuous-time process that gradually adds noise, then learns to reverse this process by estimating the score function at each noise level. Crucially, the denoising step at each noise level can be optimized independently from other noise levels. This independence property provides the theoretical foundation that has been missing from block-wise training approaches: it allows us to partition networks into blocks, each responsible for a specific noise level range, without compromising global coherence.

### Our approach: interpreting networks as diffusion processes

We propose **DiffusionBlocks**, a framework that enables principled block-wise training by interpreting sequential layer updates in transformer-based networks as discretized steps of a continuous-time diffusion process. Building on the established connection between residual networks and differential equations, we leverage the fact that residual connections naturally correspond to Euler discretization of the probability flow ODE in diffusion models. This correspondence allows us to partition networks with residual connections, particularly transformer-based networks, into blocks that each handle specific noise-level ranges. These blocks can be trained completely independently, requiring gradients for only one block at a time.

Figure 1 illustrates the core concept of DiffusionBlocks. Unlike previous block-wise methods with ad-hoc objectives, our framework derives each block's training objective from score matching theory. As a result, consistent local optimization at each noise level collectively yields a faithful approximation of the global reverse process, while also allowing practitioners to seamlessly adopt techniques from Karras et al. (2022) to further enhance training.

![Overview of DiffusionBlocks](figures/DiffusionBlocks.pdf)

**Figure 1. Overview of DiffusionBlocks.**
**Left:** Standard networks require backpropagation through all layers.
**Center:** DiffusionBlocks partitions networks into blocks, each trained independently to denoise within assigned noise ranges.
**Right:** Applications. For diffusion models (top), inference requires only the relevant block per denoising step. For recurrent-depth models (bottom), our framework replaces iterative training with single-pass training, eliminating the computational overhead of backpropagation through time.

Our main contributions are:

- **Block-wise training via continuous-time diffusion interpretation:** We show that transformer-based networks can be interpreted as implementing discretized steps of continuous-time diffusion processes (Section 2.2), enabling genuinely independent block training. Each block learns to denoise within its assigned noise level range, requiring gradients for only one block at a time during training (Section 3.1).

- **Equi-probability partitioning for balanced learning**: We propose a principled, diffusion theoretic strategy that partitions noise levels based on equal cumulative probability mass, ensuring balanced parameter utilization across blocks (Section 3.3).

- **Broad applicability with maintained performance:** We conduct extensive experiments (Section 5), demonstrating that DiffusionBlocks successfully applies to diverse architectures (vision, diffusion, autoregressive, recurrent-depth, and masked diffusion), achieving competitive performance to end-to-end backpropagation while requiring gradients for only one block at a time. Additionally, our framework naturally extends to recurrent-depth models, transforming their multiple-iteration training into single-pass training (Section 5.4).

- **Significant efficiency gains:** During training, only one block requires gradient computation, reducing memory requirements proportionally to the number of blocks. For diffusion models, inference requires only one relevant block per denoising step (Section 5.2). For recurrent-depth models, our framework eliminates _K_ iterations during training, demonstrating up to _K_-fold reduction in training computation (Section 5.4).

---

## 2. Preliminaries

### 2.1 Score-based diffusion models

We adopt the Variance Exploding (VE) formulation where a clean data **y** ~ p_data is perturbed with Gaussian noise at noise level σ:

$$\mathbf{z}_{\sigma} = \mathbf{y} + \sigma \boldsymbol{\epsilon} \quad \text{where} \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

For generations, we use the deterministic probability flow ODE that reverses the noising process:

$$\frac{\mathrm{d}\mathbf{z}_{\sigma}}{\mathrm{d}\sigma} = -\sigma \nabla_{\mathbf{z}} \log p_{\sigma}(\mathbf{z}_{\sigma}) \tag{1}$$

where ∇***z** log p*σ(**z**_σ) is the score function. Using Tweedie's formula, the score is approximated via a denoiser D_**θ**(**z**\_σ, σ) that predicts clean data from noisy input:

$$\nabla_{\mathbf{z}} \log p_{\sigma}(\mathbf{z}_{\sigma}) \approx \frac{D_{\boldsymbol{\theta}}(\mathbf{z}_{\sigma}, \sigma) - \mathbf{z}_{\sigma}}{\sigma^2}$$

The denoiser is trained by minimizing:

$$\mathcal{L}(\boldsymbol{\theta}) := \mathbb{E}_{\mathbf{z}_0 \sim p_{\text{data}}, \sigma \sim p_{\text{noise}}, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ w(\sigma) \|D_{\boldsymbol{\theta}}(\mathbf{y} + \sigma\boldsymbol{\epsilon}, \sigma) - \mathbf{y}\|_2^2 \right] \tag{2}$$

where w(σ) weights different noise levels and p_noise is the noise level distribution used during training. The choice of p_noise determines which noise levels are emphasized during training. Karras et al. (2022) use a log-normal distribution to concentrate training on perceptually important intermediate noise levels where image structure emerges. The weighting w(σ) is designed to counteract the sampling bias from p_noise, ensuring balanced gradient magnitudes across all noise levels.

### 2.2 Residual connections as Euler steps of the reverse diffusion process

The connection between residual networks and differential equations has been established in prior works. We extend this perspective to show that residual networks naturally implement discretized steps of the reverse diffusion process.

Applying Euler discretization to Eq. (1) with noise levels σ*0 > σ_1 > ··· > σ_T, we define Δσ*ℓ := σ*{ℓ−1} − σ*ℓ > 0 and obtain:

$$\mathbf{z}_{\sigma_l} = \mathbf{z}_{\sigma_{l-1}} - \Delta\sigma_\ell \cdot \sigma_{\ell-1} \nabla_{\mathbf{z}}\log p_{\sigma_{\ell-1}}(\mathbf{z}_{\sigma_{\ell-1}})$$

$$= \mathbf{z}_{\sigma_{\ell-1}} + \frac{\Delta\sigma_\ell}{\sigma_{\ell-1}} \left(\mathbf{z}_{\sigma_{\ell-1}} - D_{\boldsymbol{\theta}}(\mathbf{z}_{\sigma_{\ell-1}}, \sigma_{\ell-1})\right) \tag{3}$$

Modern architectures such as Transformers employ residual connections where each block updates its input through an additive transformation:

$$\mathbf{z}_{\ell} = \mathbf{z}_{\ell-1} + f_{\theta_\ell}(\mathbf{z}_{\ell-1})$$

where **z**_ℓ ∈ ℝ^d denotes the intermediate output of block ℓ, and f_{θ*ℓ} is the block transformation parameterized by θ*ℓ. This structure appears in ResNets, Transformers, and other modern architectures.

This scheme is also used in the recent development of recurrent-depth models, which apply the same network parameters **θ** recursively K times: **z**_k = **z**_{k−1} + f***θ**(**z***{k−1}) for k ∈ [K]. However, these methods suffer from the expensive _backpropagation through time (BPTT)_, and various measures have been taken to reduce its computational burden.

The critical observation is that, in the setting of the diffusion introduced in the previous section, D\_**θ** in Eq. (3) can be trained with Eq. (2) without BPTT, thereby providing a theoretically sound optimization method of a dynamical system through an ensemble of local optimization. In the next section, we provide a recipe for converting networks with skip connections into diffusion, thereby replacing the _backpropagation through layers_ with the optimization scheme analogous to Eq. (2).

---

## 3. Method

![3-step conversion of a standard neural network to DiffusionBlocks at training phase](figures/meta-algo.pdf)

**Figure 2. 3-step conversion of a standard neural network to DiffusionBlocks at training phase.**
**Step 1:** Partition L layers into B blocks.
**Step 2:** Define noise distribution p*σ (e.g., log-normal) and partition the range [σ_min, σ_max] into B intervals {[σ_b, σ*{b−1}]}*{b=1}^B, assigning each block a specific noise range (Section 3.3).
**Step 3:** Augment blocks with noise conditioning: extend input to x̃ = (**x**, **z***σ) where **z**\_σ = **y** + σ**ε**, and incorporate noise-level conditioning (e.g., via AdaLN). Then, each block is trained independently from other blocks to predict target **y** within its assigned noise range.

### 3.1 Converting a neural network to DiffusionBlocks

Our goal is to transform a given feedforward system into a discretized version of the recursive denoising steps in the diffusion model. Throughout this paper, we denote by (**x**, **y**) the input-output pairs where **x** represents the network input (e.g., images for classification) and **y** is the target output (e.g., class label for classification).

Consider a neural network in a form of a stack of set-to-set maps (e.g. transformer-based networks) ℱ = {f*{θ*ℓ} | ℓ ∈ [L]} with the same output and input dimensions, so that f*{θ*ℓ} maps a variable set of tokens in ℝ^d to the same number of tokens in ℝ^d. The original network processes the input with f*{θ_L} ∘ ··· ∘ f*{θ*0}, followed possibly by a readout module. Or, in more conventional formulation with the presence of residual, the original network may update the ℓ-th layer input **z***ℓ to the next layer via the rule **z***{ℓ+1} = **z***ℓ + f*{θ*ℓ}(**z**\_ℓ).

We transform this network into a stack of Diffusion Blocks through the following three steps (Figure 2).

**Step 1: Block partitioning.**
We partition ℱ into B blocks ℱ = ⊎*{b=1}^B ℱ_b, where ℱ_b contains layers indexed by {ℓ*{b−1}+1, …, ℓ*b}. Let f̄*{**θ**_b} := f_{θ*{ℓ_b}} ∘ ··· ∘ f*{θ*{ℓ*{b-1}+1}} be the composition of layers in ℱ_b.

**Step 2: Noise range assignment.**
We define a noise distribution p*noise and define a noise range [σ_min, σ_max]. We partition the range into B intervals {[σ_b, σ*{b−1}]}\_{b=1}^B. We recommend the choice of log-normal for p_noise, following Karras et al. (2022), along with the partitioning strategy in Section 3.3.

**Step 3: Augmenting blocks with noise conditioning.**
Finally, we suit {f̄*{θ_b}}\_b to the update rule in Eq. (3) by letting f̄*{**θ**_b} play the role of D_{**θ**_b}. Leveraging the assumption that f̄_{**θ**_b} is a map from a set of tokens to a set of tokens, we alter the input f̄_{**θ**_b} from **x** to x̃ = (**x**, **z**). Additionally, we extend each block f_{**θ**_b} to incorporate noise-level conditioning through, for example, via normalization (AdaLN). We denote this noise-conditioned version as f̄_{**θ**\_b|σ}. Altogether, the update of the diffusion block constructed from ℱ is given by:

$$\mathbf{z}_b = \mathbf{z}_{b-1} + \frac{\Delta\sigma_b}{\sigma_{b-1}} \left(\mathbf{z}_{b-1} - [\bar{f}_{\boldsymbol{\theta}_b | \sigma_{b-1}}(\mathbf{x}, \mathbf{z}_{b-1})]_\mathbf{z} \right) \tag{4}$$

where [f̄(·)]_**z** is the set of tokens corresponding to **z**. More abstractly put, our modified update rule Eq. (4) can be rewritten as **z**\_b = α**z**_{b−1} + βf̄*{**θ**\_b|σ*{b−1}}(**x**, **z**\_{b−1}) where α and β are constants dependent on σ ratio.

At the time of inference, **z**\_b serves as the intermediate estimator of the target variable, with **z**\_0 = σ_max**ε** being the pure noise. Please see Figure A2 in Appendix B for the conversion of this inference process.

### 3.2 Block-independent training of the diffusion blocks

By the network modification recipe in the previous section, we transform the original feedforward map to the recursive denoising map in a diffusion process. The advantage of this modification is the fact that the objective in Eq. (2) can be optimized at any noise level σ independently without knowledge of other noise levels. This allows us to define a training objective for each block b:

$$\mathcal{L}_b(\boldsymbol{\theta}_b) := \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim p_{\text{data}}, \sigma \sim p_{\text{noise}}^{(b)}, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ w(\sigma)\cdot \text{Loss}( \bar{f}_{\boldsymbol{\theta}_b | \sigma}(\mathbf{x}, \mathbf{y} + \sigma \boldsymbol{\epsilon}) , \mathbf{y})\right] \tag{5}$$

where p*noise^{(b)} is the noise distribution p_noise with the support of [σ_b, σ*{b−1}] and renormalized, and Loss(·, ·) is the inner loss function, typically L2 loss as in Eq. (2).

Each block independently learns to denoise within its assigned range, with training samples drawn according to the original distribution p*noise. Collectively, the B blocks cover the entire noise distribution: ∪*{b=1}^B [σ_b, σ_{b−1}] = [σ_min, σ_max], ensuring that the complete network can denoise at any noise level while each block specializes in its designated range.

This independence enables training with memory requirements for only L/B layers, storing activations only for the active block, compared to all L layers required by standard training. This approach achieves a B× memory reduction during training, as gradients are computed for only one block at a time.

**Figure 3: Training and inference algorithms.**

|               | Standard Network                                              | DiffusionBlocks                                                                                     |
| ------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Training**  | Given: Network with parameters **θ**                          | Given: A single block b ∈ [B] with parameters **θ**\_b                                              |
|               | Sample data (**x**, **y**)                                    | Sample data (**x**, **y**)                                                                          |
|               | **z**\_0 ← **x**                                              | Sample σ ~ p*noise^{(b)} from [σ_b, σ*{b−1}]                                                        |
|               | For ℓ = 1 to L: **z**_ℓ ← **z**_{ℓ−1} + f*{θ*ℓ}(**z**\_{ℓ−1}) | ŷ ← f̄\_{**θ**\_b\|σ}(**x**, **y** + σ**ε**), **ε** ~ N(**0**, **I**)                                |
|               | ŷ ← **z**\_L                                                  | ℒ ← w(σ) · Loss(ŷ, **y**)                                                                           |
|               | ℒ ← Loss(ŷ, **y**)                                            | Update only **θ**\_b via backprop                                                                   |
|               | Update all **θ** via backprop                                 |                                                                                                     |
| **Inference** | Input: **x**                                                  | Input: **x**, noise levels {σ*i}*{i=1}^T                                                            |
|               | **z**\_0 ← **x**                                              | **z**\_0 ~ N(**0**, σ_max^2 **I**)                                                                  |
|               | For ℓ = 1 to L: **z**_ℓ ← **z**_{ℓ−1} + f*{θ*ℓ}(**z**\_{ℓ−1}) | For i = 0 to T−1: Select block b where σ*i ∈ [σ_b, σ*{b−1}]                                         |
|               | Output: **z**\_L                                              | ŷ ← f̄*{**θ**\_b\|σ*{i−1}}(**x**, **z**_{i−1}); **z**\_i ← Euler step(**z**_{i−1}, ŷ, σ\_{i−1}, σ_i) |
|               |                                                               | Output: **z**\_T                                                                                    |

### 3.3 Block partitioning strategy

A critical design choice in DiffusionBlocks is how to partition the noise level range [σ_min, σ_max] into B intervals. A naive approach would divide the range uniformly: σ_b = σ_min + b·(σ_max − σ_min)/B. However, this fails to account for the varying difficulty of denoising at different noise levels.

Following Karras et al. (2022), we adopt a log-normal distribution for sampling noise levels during training: log σ ~ N(P_mean, P_std^2). This distribution concentrates probability mass at intermediate noise levels, which empirically contribute most to generation quality.

![Equi-probability partitioning (B=3)](figures/boundaries.pdf)

**Figure 4. Equi-probability partitioning (B=3).** Blocks partition the log-normal p_σ by equal probability mass (orange boundaries), not uniform spacing (gray), concentrating capacity where denoising is most challenging.

To preserve this distribution across the entire network while ensuring each block handles equal denoising difficulty, we partition based on cumulative probability mass. Specifically, we choose boundaries {σ*b}*{b=1}^B such that each block handles exactly 1/B of the total probability mass:

$$\int_{\sigma_{b-1}}^{\sigma_b} p_{\text{noise}}(\sigma) d\sigma = 1/B$$

The block boundaries are computed as σ*b = exp(P_mean + P_std · Φ^{−1}(q_b)), where Φ^{−1} is the inverse standard normal CDF and q_b = q_min + (b/B)(q_max − q_min), with q*{min/max} = Φ((log σ\_{min/max} − P_mean)/P_std).

This _equi-probability partitioning_ ensures that each block handles an equal amount of the training distribution's probability mass, leading to balanced parameter utilization. As shown in Figure 4, blocks assigned to intermediate noise levels, where denoising is most challenging, receive narrower intervals, while blocks handling very high or low noise levels receive wider intervals.

---

## 4. Related Works

### Block-wise training methods

Various block-wise training approaches partition networks into independently trainable components but lack theoretical grounding, relying on heuristic objectives that fail to guarantee global performance when optimized locally. Approaches like Forward-Forward algorithm rely on contrastive objectives, which fundamentally limit them to classification tasks and make adaptation to generation non-trivial. In contrast, DiffusionBlocks leverages denoising score matching theory, which naturally decomposes into independent local objectives without task-specific constructs, enabling application to both classification and generative tasks.

### Comparison with NoProp

Concurrently with our submission, Li et al. (2025) also released a backpropagation-free strategy in close relation to our philosophy. However, they present their technique together with the custom CNN-based architecture in one package and evaluate only on classification tasks, making it unclear how to apply their approach to modern architectures or tasks other than the classification they showcase in their work. In contrast, DiffusionBlocks provides a systematic procedure for converting any residual networks, particularly modern transformers, into block-wise trainable models with minimal modifications. We partition the continuous noise range using equi-probability partitioning and demonstrate success on both generative tasks and classification tasks. In Section 5.5, we apply DiffusionBlocks to their architecture, and demonstrate that our continuous-time block-wise training with equi-probability partitioning is more effective.

### Stage-specific diffusion models

Several works train specialized models for different noise levels in diffusion. However, these approaches train models jointly or fine-tune from shared parameters. DiffusionBlocks trains blocks independently, with no shared parameters or joint fine-tuning, achieving complete isolation.

---

## 5. Experimental Results

We evaluate DiffusionBlocks across diverse architectures and tasks to demonstrate its generality and effectiveness. Detailed experimental configurations are provided in Appendix D. For each architecture, we report task performance alongside the memory reduction factor B, where only L/B layers require gradients during training.

**Baselines.** Because DiffusionBlocks is a framework for transforming networks into block-wise trainable models, we evaluate its efficacy by comparing the modified network (trained block-wise) against the original network (trained with end-to-end backpropagation). Other block-wise training methods in practice today also include Forward-Forward (FF) and the concurrent NoProp. Fair comparison against these methods warrants careful experimental design. Firstly, we compare against FF only on classification tasks (Section 5.1) since its contrastive objective does not naturally extend to generation. Also, because NoProp is proposed together with a custom architectural design rather than with a principled transformation procedure to be applied to a vanilla network, the adaptation of NoProp to other architectures involves nontrivial design choices and freedom. To enable fair comparison with NoProp, we therefore use their specific architecture as the base diffusion model on which to apply our DiffusionBlocks (Section 5.5).

### 5.1 Vision transformers for image classification

We first validate DiffusionBlocks on classification tasks using Vision Transformer (ViT) on CIFAR-100. A 12-layer ViT is partitioned into B=3 blocks, with noise added to class label embeddings during training. We compare against the Forward-Forward algorithm, a representative block-wise training method that uses contrastive objectives.

**Table 1. ViT results on CIFAR-100.** DiffusionBlocks achieves comparable accuracy while training only 4 layers at a time, outperforming Forward-Forward algorithm.

| Method                | Accuracy (↑) |
| --------------------- | ------------ |
| ViT                   | 60.25        |
| + Forward-Forward     | 7.85         |
| **+ DiffusionBlocks** | **59.30**    |

DiffusionBlocks maintains baseline accuracy while requiring gradients for only 4 layers. Notably, Forward-Forward achieves only 7.85% accuracy, highlighting the importance of principled denoising objectives over ad-hoc contrastive approaches.

### 5.2 Diffusion models for image generation

Having established its effectiveness on classification tasks, we now turn to generative models. We apply DiffusionBlocks to DiT within the EDM framework. We evaluate 12-layer DiT (DiT-S/2) on CIFAR-10 and 24-layer DiT (DiT-L/2) on ImageNet at 256×256 resolution, both with B=3 blocks. During inference, we use Euler sampling with 50 steps and classifier-free guidance (scale 2.0).

**Table 2. DiT results for image generation.** FID is computed on both training and test splits (train / test). DiffusionBlocks achieves comparable scores while reducing training memory and inference cost by 3×.

| Dataset  | Method                | FID (↓)           |
| -------- | --------------------- | ----------------- |
| CIFAR-10 | DiT                   | 32.84 / 39.83     |
| CIFAR-10 | **+ DiffusionBlocks** | **30.59 / 37.20** |
| ImageNet | DiT                   | 9.01 / 12.09      |
| ImageNet | **+ DiffusionBlocks** | **9.00 / 10.63**  |

DiffusionBlocks achieves comparable FID scores with 3× memory reduction. Additionally, inference requires only one block per denoising step, providing computational savings proportional to the number of steps.

### 5.3 Masked diffusion models for text generation

We extend DiffusionBlocks to masked diffusion language models using MD4 on the text8 dataset. While continuous diffusion models naturally map to our framework through noise levels σ, extending DiffusionBlocks to discrete masked diffusion requires careful adaptation. Specifically, we partition the masking schedule rather than continuous noise levels, ensuring each block handles an equal share of the demasking work (details in Appendix C). We use a 12-layer DiT-based transformer partitioned into B=3 blocks.

**Table 3. Masked diffusion model (MDM) results on text8.** DiffusionBlocks improves BPC while training with 3× less memory through masking schedule partitioning.

| Method                | BPC (↓)  |
| --------------------- | -------- |
| MDM                   | 1.56     |
| **+ DiffusionBlocks** | **1.45** |

DiffusionBlocks achieves 1.45 bits-per-character (BPC) compared to MD4's 1.56, while using 3× less memory. This improvement confirms that our principled noise-level partitioning effectively extends to discrete diffusion processes.

### 5.4 Autoregressive models for text generation

**Table 4. Autoregressive (AR) transformer results for text generation.** DiffusionBlocks maintains generation quality with 4× memory reduction on both LM1B and Openwebtext (OWT) datasets.

| Dataset | Method                | MAUVE (↑) | PPL (Llama-2) (↓) | PPL (GPT2-XL) (↓) |
| ------- | --------------------- | --------- | ----------------- | ----------------- |
| LM1B    | AR                    | 0.50      | 14.58             | 38.87             |
| LM1B    | **+ DiffusionBlocks** | **0.71**  | **12.32**         | **30.99**         |
| OWT     | AR                    | **0.85**  | 15.05             | **25.24**         |
| OWT     | **+ DiffusionBlocks** | 0.82      | **14.99**         | 26.33             |

We demonstrate that DiffusionBlocks successfully transforms standard autoregressive (AR) models, which are architectures originally designed for next-token prediction, not denoising. Using 12-layer Llama-2-style transformers with B=4 blocks, we evaluate on 1 Billion Words Dataset (LM1B) and OpenWebText (OWT). While AR models are typically evaluated using perplexity, computing traditional perplexity is non-trivial for our diffusion framework as it is not derived from ELBO. Instead, we evaluate using MAUVE scores following SEDD to measure similarity between generated and real text. We also report generative perplexity from two teacher models, Llama-2-7B and GPT2-XL.

DiffusionBlocks achieves comparable performance despite training only 3 layers at a time, demonstrating the framework's broad applicability beyond diffusion-native architectures.

### 5.5 Recurrent-depth models for text generation

**Table 5. Recurrent-depth model results for text generation.** DiffusionBlocks eliminates 32 training iterations, achieving better performance with single-pass training.

| Method                | MAUVE (↑) | PPL (Llama-2) (↓) | PPL (GPT2-XL) (↓) |
| --------------------- | --------- | ----------------- | ----------------- |
| Huginn                | 0.49      | 17.04             | 46.73             |
| **+ DiffusionBlocks** | **0.70**  | **16.08**         | **42.43**         |

We showcase a different application of DiffusionBlocks beyond block-wise training. The updates in recurrent-depth models naturally correspond to diffusion steps. Following Section 3.1, we apply DiffusionBlocks to Huginn, a recurrent-depth model that applies the same network multiple times, starting from noise. While Huginn uses 8-step truncated BPTT to avoid the full BPTT over 32 iterations, DiffusionBlocks makes this optimization even more efficient, because it only requires a single forward pass per training step. DiffusionBlocks shows better performance on LM1B for text generation while eliminating 32 iterations. This demonstrates that our framework enables fundamental training transformations beyond block-wise training.

### 5.6 Analysis

#### 5.6.1 Comparison with NoProp

We compare DiffusionBlocks with NoProp as an ablation study, applying to their custom CNN-based architecture to isolate the effect of our continuous-time block-wise training using equi-probability partitioning.

**Table 6. Comparison with NoProp on CIFAR-100.** DiffusionBlocks achieves both continuous-time formulation and layer-wise training. All scores except DiffusionBlocks are taken from NoProp. Note that the Backprop result is from applying BPTT to the sampled paths of a specific form of SDE resembling NoProp-DT.

| Method                     | Continuous | Block-wise | Accuracy (↑) |
| -------------------------- | ---------- | ---------- | ------------ |
| Backprop                   |            |            | 47.80        |
| NoProp-DT                  |            | ✓          | 46.06        |
| NoProp-CT                  | ✓          |            | 21.31        |
| NoProp-FM                  | ✓          |            | 37.57        |
| **(Ours) DiffusionBlocks** | **✓**      | **✓**      | **46.88**    |

DiffusionBlocks outperforms all NoProp variants. Notably, while maintaining comparable performance to the backpropagation, DiffusionBlocks is the only method that successfully combines continuous-time formulation with block-wise training. This demonstrates that our equi-probability partitioning with independent denoisers per block is crucial for continuous-time block-wise training.

#### 5.6.2 Ablation Studies on Design Choices

**Table 7. Effect of block partitioning strategy on CIFAR-10.** Layer distribution indicates the number of layers in each of the 3 blocks (totaling 12 layers).

| Partitioning Strategy | Layer Distribution | FID (↓)   |
| --------------------- | ------------------ | --------- |
| Uniform               | [4,4,4]            | 43.53     |
| Uniform               | [3,6,3]            | 43.59     |
| Uniform               | [6,4,2]            | 47.49     |
| Uniform               | [2,4,6]            | 42.37     |
| **Equi-Probability**  | **[4,4,4]**        | **38.03** |
| Equi-Probability      | [3,6,3]            | 41.64     |
| Equi-Probability      | [6,4,2]            | 45.42     |
| Equi-Probability      | [2,4,6]            | 40.40     |

**Table 8. Effect of block count on ImageNet.** Fewer blocks achieve better FID but require more layers per diffusion step, creating a trade-off between quality and efficiency. Scores surpassing end-to-end backpropagation (B=1) are in **bold**.

| Number of Blocks | FID (↓)   | L/B (↓) | Relative Speed |
| ---------------- | --------- | ------- | -------------- |
| B=1              | 12.09     | 24      | 1.0×           |
| **B=2**          | **9.90**  | 12      | 2.0×           |
| **B=3**          | **11.11** | 8       | 3.0×           |
| **B=4**          | **11.90** | 6       | 4.0×           |
| B=6              | 14.43     | 4       | 6.0×           |

**Block partitioning strategy.** Equi-probability partitioning achieves significantly better FID across all layer distributions. The improvement stems from allocating computational resources based on denoising difficulty: equi-probability assigns more blocks to challenging intermediate noise levels where most learning occurs, while uniform partitioning wastes capacity on trivial very high/low noise regions. Notably, within equi-probability partitioning, uniform layer distribution (4-4-4) achieves the best FID, demonstrating that practitioners can simply divide layers equally without tuning since the noise-based partitioning automatically balances learning difficulty across blocks.

**Number of blocks B.** The table reveals the trade-off between generation quality and efficiency on ImageNet. Notably, moderate block counts (B=2 or B=3) achieve better FID than end-to-end training (B=1), suggesting that moderate block partitioning can actually improve performance through specialization. As B increases further, quality gradually declines due to reduced capacity per block, though inference speed improves linearly.

---

## 6. Conclusion

We introduced DiffusionBlocks, a theoretically grounded framework that transforms residual networks into independently trainable blocks through continuous-time diffusion interpretation. By recognizing that residual connections naturally implement discretized diffusion steps, we provide a systematic recipe requiring minimal modifications that maintains competitive performance across diverse architectures while achieving B× memory reduction during training.

### Future Works

Our work opens several important directions for future research. First, while we consistently used Euler discretization to match residual connections, other diffusion samplers could be employed within blocks with modified inter-block connections. Second, DiffusionBlocks currently requires matching input-output dimensions, which limits its application to architectures like U-Net. Third, while we demonstrate DiffusionBlocks' effectiveness on models trained from scratch, scaling to even larger models would further demonstrate its practical impact. Particularly, a promising direction is to convert pre-trained large models to DiffusionBlocks through fine-tuning rather than training from scratch.

Fourth, determining the optimal granularity of block partitioning presents an interesting theoretical and practical challenge. While our experiments demonstrate that treating entire architectural blocks (e.g., complete ViT blocks) as single denoising units works well, a principled method for selecting the ideal partitioning granularity based on architecture and task characteristics could further enhance the framework's applicability.

Finally, understanding why moderate block partitioning sometimes outperforms end-to-end training warrants theoretical investigation. We hypothesize two contributing factors: (1) DiffusionBlocks employs a different optimization structure in which each block is directly linked to the target through a denoising objective, creating a learning signal that differs from standard end-to-end training; and (2) assigning different noise ranges to different blocks may induce beneficial specialization effects. Combined with equi-probability partitioning, this introduces a natural form of curriculum learning by allocating balanced difficulty across blocks.

DiffusionBlocks represents a step toward democratizing large-scale model training by reducing computational requirements without sacrificing performance, making advanced AI capabilities more accessible.

### Author Contributions

Makoto Shing conceptualized the DiffusionBlocks framework, developed its diffusion-theoretic formulation connecting residual networks and continuous-time diffusion processes, implemented the method, conducted all experiments, and wrote the manuscript. Masanori Koyama provided theoretical insights into the diffusion-based interpretation and contributed to refining both the manuscript and the conceptual positioning of the work. Takuya Akiba supervised the research and provided technical guidance and feedback throughout the project. All authors contributed to the interpretation of results and manuscript revision.

### Acknowledgement

The authors would like to thank Stefano Peluchetti for helpful feedback on an earlier version of the draft.

---

## Appendix A: Notations

| Notation           | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| x ~ 𝒳              | Conditioning/Input to the network (task-dependent)           |
| y ∈ 𝒴              | Clean target data (task-dependent)                           |
| σ ∈ ℝ              | Noise level in continuous diffusion                          |
| z_σ ∈ ℝ^d          | Noisy data at noise level σ: z_σ = y + σε, where ε ~ N(0, 1) |
| z_ℓ ∈ ℝ^d          | Intermediate activation at layer/block ℓ                     |
| D_θ: ℝ^d × ℝ → 𝒴   | Denoiser network with parameters θ                           |
| f*{θ*ℓ}: ℝ^d → ℝ^d | Layer/block transformation with parameters θ_ℓ               |
| B                  | Number of blocks                                             |
| L                  | Total number of layers                                       |

**Examples of (x, y) on a task:**

| Task                     | x                                      | y                 |
| ------------------------ | -------------------------------------- | ----------------- |
| Image classification     | input image                            | class label       |
| Image generation         | noisy image (optionally + class label) | clean image       |
| Text Generation (AR)     | previous tokens                        | next token        |
| Text Generation (Masked) | sequence with mask tokens              | unmasked sequence |

---

## Appendix B: Extension to Diverse Architectures

While we have described DiffusionBlocks for standard residual networks where inputs and outputs naturally live in the same d-dimensional space, the framework extends to specialized architectures.

![Converting different architectures to DiffusionBlocks: Training](figures/meta-algo-training.pdf)

**Figure A1. Converting different architectures to DiffusionBlocks: Training.** During training, noise is added to target outputs (labels, embeddings, or images) and each block learns to denoise within its assigned noise range. Blocks are sampled randomly and trained independently, requiring gradients for only one block at a time.

![Converting different architectures to DiffusionBlocks: Inference](figures/meta-algo-inference.pdf)

**Figure A2. Converting different architectures to DiffusionBlocks: Inference.** During inference, blocks are applied sequentially from σ_max to σ_min. The figure shows the first denoising step where block b=1 transforms pure noise **z**\_0 into the next state **z**\_1. Only the relevant block is active at each noise level, providing memory efficiency. ⊕ denotes the Euler step in Eq. (4).

**Vision Transformers (ViT) in classification tasks.** We adapt DiffusionBlocks by adding noise to the class label embeddings while maintaining the standard ViT architecture. Specifically, we create the input sequence by concatenating the `[CLS]` token, patch embeddings **x**, and the noisy label embedding **z***σ, where **z***σ = **y**\_emb + σ**ε** and **y**\_emb ∈ ℝ^d is the learnable continuous embeddings for the class label y. Each block b learns to denoise this label representation conditioned on the patch embeddings **x**. The training loss is the standard cross-entropy between the classification head's output logits (applied to the `[CLS]` token) and the true class labels.

**For diffusion models**, DiffusionBlocks provides a natural fit: these models already operate by denoising, so partitioning simply assigns different noise ranges to different blocks without architectural modifications. The standard denoiser D***θ**(**z***σ, σ) becomes D*{**θ**\_b}(**z***σ, σ) for block b.

**For discrete output spaces like language modeling**, we operate in the embedding space. Noise is added after the embedding layer: given input tokens **x**, we compute **z** = f*in(**x**), then add noise **z***σ = **z** + σ**ε**. For autoregressive models, the denoiser D*{**θ**\_b}(**z***{i,σ}, **z**_{<i}, σ) recovers the clean embedding of token i from its noisy version, conditioned on previous clean token embeddings **z**_{<i}. We minimize cross-entropy loss instead of L2 loss.

**For recurrent-depth architectures** that apply the same network K times, we interpret the entire recurrence as a diffusion process. Instead of training with K forward passes through recurrent iterations, we train the network as a denoiser D*θ(**z***σ, **x**, σ) by sampling σ ~ p_σ and performing a single forward pass to map noisy input to clean output, reducing computational cost by factor K while maintaining the original K-iteration inference procedure.

---

## Appendix C: Implementation Details in DiffusionBlocks

**Overlap between blocks.** To smooth transitions across block boundaries, we slightly extend each block's noise interval in log-σ space. For a block b responsible for [σ_b, σ_{b−1}] with σ*{b−1} > σ_b, we define α_b := (σ*{b−1}/σ*b)^γ, where γ ≥ 0, and train over the expanded range [σ_b/α_b, α_b σ*{b−1}]. Here γ controls the degree of overlap: γ=0 recovers non-overlapping intervals, while γ>0 yields smoother transitions between blocks. In practice, we found γ ∈ [0.0, 0.1] effective, and we use 0.05 by default and 0.1 for text generation.

**Weighting and preconditioning.** Following the EDM framework, we use the weighting function:

$$w(\sigma) = \frac{\sigma^2 + \sigma_{\text{data}}^2}{(\sigma \cdot \sigma_{\text{data}})^2}$$

where σ*data = 0.5 for all experiments. The weighting is crucial for equi-probability partitioning to work effectively, as it counteracts the sampling bias introduced by the log-normal distribution p*σ. We also adopt EDM's preconditioning scheme, which involves input scaling to ensure stable training dynamics across all noise levels.

**Normalizing embeddings.** For tasks where the target variables are discrete (e.g. class labels in image classification or token ids in text generation), DiffusionBlocks operates the diffusion process in the continuous embedding space. A known issue in continuous relaxation of discrete variables is _embedding collapse_, where all learned embeddings correspond to the same vector. To prevent this, we apply L2 normalization to the embeddings following Dieleman et al. (2022).

**Training and inference details.** For training efficiency, blocks are randomly sampled per iteration, requiring memory for only L/B layers. Blocks can alternatively be trained in parallel across multiple GPUs when available. During inference, we generate samples by sequentially applying blocks from σ_max to σ_min. While we use Euler steps in our experiments due to the natural correspondence between residual connections and Euler discretization, our framework is not limited to this choice.

---

## Appendix D: Masked Diffusion Language Models as DiffusionBlocks

### D.1 Continuous-time formulation

We first recall the continuous-time formulation of masked diffusion language models. Let **x**_0 = (x_{01},…,x\_{0n}) denote a sequence of tokens and let α(t): [0,1] → [1,0] denote the masking schedule at continuous time t ∈ [0,1], where α(t) represents the probability of remaining unmasked. The forward process progressively masks tokens as:

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \prod_{i=1}^n q(x_{ti} \mid x_{0i}) \quad\text{where}\quad x_{ti} = \begin{cases} x_{0i}, & \text{with prob. } \alpha(t),\\ \texttt{[MASK]}, & \text{with prob. } 1-\alpha(t). \end{cases}$$

The training objective in continuous form is:

$$\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{\mathbf{x}_0} \int_0^1 \frac{-\alpha'(t)}{1-\alpha(t)} \mathbb{E}_{\mathbf{x}_t \sim q(\mathbf{x}_t\mid \mathbf{x}_0)} \left[ \sum_{i : x_{ti}=\texttt{[MASK]}} \text{CE}\!\left(f_\theta(\mathbf{x}_t, t)_i, x_{0i}\right) \right] dt$$

### D.2 Partitioning into DiffusionBlocks

To enable block-wise training, we partition the objective into B disjoint intervals in t. The contribution of interval [t_a, t_b] is:

$$\int_{t_a}^{t_b} -\alpha'(t)\,dt = \alpha(t_a)-\alpha(t_b)$$

This shows that the training mass is distributed uniformly in α, not in t. Therefore, the natural partition boundaries are defined by equal decrements of α:

$$\alpha_b = 1 - \tfrac{b}{B}, \quad b=0,\dots,B$$

with corresponding time boundaries obtained by inversion: t_b = α^{−1}(1 − b/B). For a linear schedule α(t) = 1−t, this simply yields t_b = b/B.

Each block b is then trained independently on its assigned interval:

$$\mathcal{L}_b(\boldsymbol{\theta}_b) = \mathbb{E}_{\mathbf{x}_0} \int_{t_{b-1}}^{t_b} \frac{-\alpha'(t)}{1-\alpha(t)} \mathbb{E}_{\mathbf{x}_t\sim q(\mathbf{x}_t\mid \mathbf{x}_0)} \left[\sum_{i:\,x_{ti}=\texttt{[MASK]}} \text{CE} \left(D_{\boldsymbol{\theta}_b}(\mathbf{x}_t,t)_i,\,x_{0i}\right)\right] dt \tag{A1}$$

where D*{**θ**\_b} denotes the denoiser assigned to block b. The global loss decomposes as ℒ = Σ*{b=1}^B ℒ_b.

This derivation shows that DiffusionBlocks in masked diffusion models amounts to partitioning the masking schedule α(t) rather than time. Each block is responsible for an equal decrement in α(t), i.e. an equal share of the total "demasking work", which ensures balanced parameter utilization and true independence across blocks. This construction is directly analogous to the equi-probability partitioning in continuous diffusion models described in Section 3.3.

---

## Appendix E: Experimental Details

Unless otherwise specified, all experiments use the following settings. For DiffusionBlocks, we adopt the EDM framework with default parameters: log-normal noise distribution with P_mean = −1.2 and P_std = 1.2, noise range [σ_min, σ_max] = [0.002, 80], and preconditioning following the recommended configuration. Inference uses Euler sampling with 50 steps unless stated otherwise. During training, blocks are sampled uniformly at random for each iteration.

### E.1 Vision transformers for image classification

For image classification experiments, we use a 12-layer ViT with patch size 4, 128 hidden dimensions, 4 attention heads, and 0.1 dropout, partitioned into B=3 blocks (4 layers each). We train for 500 epochs with batch size 128 and AdamW optimizer with learning rate 5×10^{−4}. We employ a cosine learning rate scheduler with a 10-epoch linear warmup. As data augmentation, we apply random horizontal flipping (p=0.5) and RandAugment. We add noise to the class label embeddings and concatenate them with the patch embeddings. We use an overlap ratio γ = 0.05 and perform 4 denoising steps during inference. For the Forward-Forward baseline, we adapt the Contrastive Forward-Forward (FF) implementation to ViT.

### E.2 Diffusion models for image generation

For image generation experiments, we use DiT-S/2 (12 layers) for CIFAR-10 and DiT-L/2 (24 layers) for ImageNet-256. Both models are partitioned into B=3 blocks. Training follows the EDM framework with classifier-free guidance (10% label dropout). For CIFAR-10, we train for 100 epochs with batch size 512 and AdamW optimizer with learning rate 10^{−4}. For ImageNet, we resize to 256×256 and encode images by a pre-trained VAE. We also train 100 epochs with batch size 512 and AdamW optimizer with learning rate 5×10^{−5}. Overlap ratio is set to γ=0.05.

In evaluation, we apply Euler sampling with 50 steps and classifier-free guidance (scale 2.0) on both CIFAR-10 and ImageNet experiments. FID is computed using 50,000 generated samples against the training and test sets, with the minimum of three evaluations reported following Karras et al. (2022).

### E.3 Masked diffusion models for text generation

We follow MD4's training protocol with 256 sequence length, AdamW optimizer with learning rate 3×10^{−4}, weight decay 0.03, and 2,000 linear warmup steps. Training runs for 100 epochs with batch size 256. The 12-layer DiT-based transformer uses 768 hidden dimensions and 12 attention heads, partitioned into B=3 blocks with overlap ratio γ=0.05. Masking schedule follows MD4's linear schedule. Bits-per-character (BPC) is evaluated on the text8 test set.

### E.4 Autoregressive models for text generation

We use a 12-layer Llama-2-style transformer augmented with time conditioning as in DiT with 768 hidden dimensions, 12 attention heads, and the Llama-2 tokenizer with 32K vocabulary size. The model is partitioned into B=4 blocks with an overlap ratio γ=0.1. Training uses sequence length 256 for LM1B and 3072 for OWT, batch size 256, AdamW with learning rate 3×10^{−4}, and 2500 warmup steps for 10 epochs.

Since DiffusionBlocks is not derived from ELBO-based objectives, computing traditional perplexity is non-trivial. Instead, we evaluate using MAUVE scores following SEDD. For each test sample, we generate 5 continuations of 50 tokens from 1K prompts and compute MAUVE against 1K reference samples with the scaling factor 0.2. Additionally, we report generative perplexity by computing the perplexity of generated text using teacher models (Llama-2-7B and GPT2-XL).

Applying DiffusionBlocks to autoregressive models requires maintaining causal consistency during training. When denoising future tokens, the model must condition on clean past tokens rather than noisy ones to preserve the autoregressive property. Following Block Diffusion, we implement this using sequence concatenation: noisy and clean sequences are concatenated with a modified causal attention mask that allows noisy tokens to attend to their corresponding clean past tokens while preventing information leakage.

### E.5 Recurrent-depth models

For Huginn, we use the default configuration: 2 prelude layers, 4-layer recurrent block, and 2 coda layers following Pythia-70M architecture with 512 hidden dimensions and 8 attention heads. Unlike other architectures, recurrent-depth models do not require block partitioning since the entire network is applied recurrently. Instead, we train the full network as a denoiser by sampling different noise levels σ at each training step. While baseline Huginn uses stochastic recurrence depth (average 32 iterations) with truncated BPTT (8 steps), DiffusionBlocks trains with single-pass diffusion. We train on LM1B for 15 epochs compared to Huginn's 5 epochs. Despite this, our approach uses approximately 10× less total computation since we avoid the 32× recurrent iterations during training.

### E.6 Ablation studies

All ablation studies follow the configurations described in Appendix E.2. We report FID scores on the test splits. For partitioning experiments, we test both uniform partitioning (equal intervals in log-space) and our equi-probability method. For block count experiments, we vary B from 2 to 6 while keeping total layers fixed at 12. We disabled the block overlap (γ=0.0) to isolate the effectiveness of each component.

**Comparison with NoProp.** We follow the experimental protocol of NoProp. In the absence of publicly available code, we implemented their NoProp-DT architecture augmented with time conditioning from NoProp-CT. Training follows NoProp-CT's hyperparameters with AdamW optimizer, learning rate 10^{−4}, batch size 128, and 1000 epochs on CIFAR-100. For DiffusionBlocks, we use B=3 blocks with overlap ratio γ = 0.1.

---

## Appendix F: Additional Experiments

### F.1 Image Classification Experiment on Tiny ImageNet

**Table A1. ViT results on Tiny-ImageNet.** DiffusionBlocks shows consistent performance on intermediate-scale classification dataset.

| Method                | Accuracy (↑) |
| --------------------- | ------------ |
| ViT                   | 35.32        |
| **+ DiffusionBlocks** | **36.16**    |

We conducted an additional experiment on the Tiny ImageNet dataset. This dataset consists of 200 classes, 100,000 training images with each image resized to 64×64 resolution. We trained a 12-layer Vision Transformer (ViT) with patch size 4, hidden size 768, and 12 attention heads. Both the baseline ViT and DiffusionBlocks models were trained for 100 epochs using a batch size of 256 and the AdamW optimizer with a learning rate of 10^{−4}. For DiffusionBlocks, we used B=2 blocks (each containing 6 layers). DiffusionBlocks maintains competitive performance relative to the baseline ViT, consistent with our findings on CIFAR-100.

### F.2 Effect of Block Count on CIFAR-10

**Table A2. Effect of block count on CIFAR-10.** Moderate block counts (2–3) achieve the best FID, showing consistent trends with ImageNet.

| Number of Blocks | FID (↓)   | L/B (↓) | Relative Speed |
| ---------------- | --------- | ------- | -------------- |
| B=1              | 39.83     | 12      | 1.0×           |
| **B=2**          | **35.47** | 6       | 2.0×           |
| **B=3**          | **38.03** | 4       | 3.0×           |
| B=4              | 45.43     | 3       | 4.0×           |
| B=6              | 53.32     | 2       | 6.0×           |

Smaller block counts tend to achieve better FID scores, and B=2 or B=3 provides strong performance. This trend matches the observations in Table 8.

### F.3 Effect of Block Count on Text Generation

**Table A3. Effect of block count on text generation (LM1B).** Best performance is achieved with B=4.

| Number of Blocks | MAUVE (↑) | Layers per Block (↓) | Relative Speed |
| ---------------- | --------- | -------------------- | -------------- |
| B=2              | 0.61      | 6                    | 2.0×           |
| B=3              | 0.65      | 4                    | 3.0×           |
| **B=4**          | **0.67**  | 3                    | 4.0×           |
| B=6              | 0.62      | 2                    | 6.0×           |

The optimal number of blocks differs between tasks: image generation achieves best FID with B=2 or B=3, while language modeling achieves best MAUVE with B=4. This motivated our choice of B=4 for language modeling experiments in the main paper.

---

## Appendix G: Comparison with Activation Checkpointing

DiffusionBlocks and activation checkpointing offer fundamentally different trade-offs and can be powerfully combined.

The key distinction lies in what each method reduces. Activation checkpointing reduces only activation memory, leaving parameters, gradients, and optimizer states unchanged. In contrast, DiffusionBlocks reduces all memory components by a factor of B.

To illustrate, consider an L-layer network where each layer has parameter size P and activation size A. With Adam optimizer (requiring 2P for momentum and variance), each layer needs 4P memory for parameters, gradients, and optimizer states. Standard training thus requires (4P + A)L total memory. Activation checkpointing reduces this to 4PL + A. DiffusionBlocks, by training B independent blocks, requires (4P + A)(L/B). Since L > B, combining DiffusionBlocks and activation checkpointing uses the least memory among these patterns.

Beyond memory reduction, DiffusionBlocks offers unique advantages regarding training time: each block can be trained in an embarrassingly parallel manner across multiple GPUs with absolutely no communication overhead.

---

## Appendix H: Training and Inference Efficiency

### Training efficiency

Consider an L-layer network trained for K iterations. Standard end-to-end backpropagation performs K × L layer evaluations. DiffusionBlocks trains only L/B layers at a time; training all B blocks for K iterations each performs (L/B) × B × K = L × K layer evaluations. Thus, DiffusionBlocks requires the _same_ total amount of computation as standard training, while reducing memory usage by a factor of B.

**Table A4. Wall-time comparison on ViT.** The aggregated DiffusionBlocks time is computed by multiplying the measured per-block iteration time by B=3.

| Method                                        | Wall time (sec/iter) |
| --------------------------------------------- | -------------------- |
| ViT                                           | 0.0507               |
| DiffusionBlocks: per-block time (4 layers)    | 0.0181               |
| DiffusionBlocks: aggregated time (0.0181 × 3) | 0.0543               |

Standard training requires 0.0507 seconds per iteration for all 12 layers. Under DiffusionBlocks with B=3, each block (4 layers) takes 0.0181 seconds per iteration (measured). The total per-iteration wall time for DiffusionBlocks is therefore 0.0181 × 3 = 0.0543 seconds. The resulting end-to-end wall time is thus comparable to standard training, with the small difference attributable to the noise-level conditioning introduced during the DiffusionBlocks conversion.

### Inference efficiency

For inference, we ensure that the total amount of computation matches that of the baseline model. For a 12-layer network, the baseline performs a single forward pass through all 12 layers. Under DiffusionBlocks with B=3, we perform three denoising steps, each invoking the corresponding 4-layer block once.

For diffusion models used in image generation, the computational benefit is even more pronounced. Standard diffusion models must apply the full network for every denoising step. With 50 denoising steps, a 12-layer DiT requires 12 × 50 layer evaluations. In DiffusionBlocks, each denoising step applies only the block responsible for that noise level, which contains 4 layers when B=3. This reduces the total compute to 4 × 50, achieving a B-fold reduction in inference cost.

---

_Code available at: <https://github.com/SakanaAI/DiffusionBlocks>_
