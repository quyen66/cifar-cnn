# Chapter 4: Advanced Detection Mechanisms for Stealthy Poisoning Attacks

## 4.1 Introduction

This chapter presents two novel detection mechanisms designed to identify stealthy poisoning attacks that evade traditional gradient-based defenses. Section 4.2 introduces **Variance-based Detection**, which exploits statistical anomalies in gradient distributions. Section 4.3 presents **Cross-round Consistency Detection**, which leverages temporal behavioral patterns across training rounds. Together, these mechanisms form **Layer 1.5: Behavioral Analysis**, positioned between the clustering-based Layer 1 and the direction-based Layer 2 in our multi-layered defense architecture.

---

## 4.2 Variance-based Detection

### 4.2.1 Motivation

The A Little Is Enough (ALIE) attack [Baruch et al., 2019] generates malicious gradients using the following transformation:

$$\mathbf{g}_{mal} = \mathbf{g}_{benign} - z \cdot \sigma(\mathbf{g}_{benign}) \cdot \text{sign}(\mathbf{g}_{benign})$$

where $z$ is a scaling factor and $\sigma(\cdot)$ denotes the standard deviation. This transformation is specifically designed to produce gradients that:

1. Maintain similar $L_2$ norms to benign gradients
2. Preserve high cosine similarity with the median gradient
3. Fall within acceptance thresholds of robust aggregation rules

However, we observe that this transformation introduces a measurable side effect: **variance compression**. The subtraction of a sign-correlated term from each gradient element reduces the overall spread of values.

### 4.2.2 Theoretical Analysis

**Theorem 4.1 (Variance Reduction under ALIE Transformation).**
*Let $\mathbf{g} \in \mathbb{R}^d$ be a gradient vector with variance $\text{Var}(\mathbf{g}) = \sigma^2$. Under the ALIE transformation with parameter $z$, the variance of the transformed gradient $\mathbf{g}_{mal}$ satisfies:*

$$\text{Var}(\mathbf{g}_{mal}) < \text{Var}(\mathbf{g})$$

*Proof.* Consider the $j$-th element of the gradient:

$$g_{mal,j} = g_j - z \cdot \sigma \cdot \text{sign}(g_j)$$

The variance of the malicious gradient is:

$$\text{Var}(\mathbf{g}_{mal}) = \mathbb{E}[(g_{mal,j} - \mu_{mal})^2]$$

Since $\text{sign}(g_j)$ is positively correlated with $g_j$, the subtraction $g_j - z \cdot \sigma \cdot \text{sign}(g_j)$ reduces the magnitude of extreme values:
- For $g_j > 0$: $g_{mal,j} = g_j - z\sigma < g_j$
- For $g_j < 0$: $g_{mal,j} = g_j + z\sigma > g_j$

This compression towards zero reduces the overall variance. $\square$

### 4.2.3 Detection Methodology

**Definition 4.1 (Gradient Variance).** For a client $i$ with gradient $\mathbf{g}_i \in \mathbb{R}^d$, the gradient variance is defined as:

$$V_i = \text{Var}(\mathbf{g}_i) = \frac{1}{d} \sum_{j=1}^{d} (g_{i,j} - \bar{g}_i)^2$$

where $\bar{g}_i = \frac{1}{d}\sum_{j=1}^{d} g_{i,j}$ is the mean gradient value.

**Definition 4.2 (Variance Ratio).** For a set of participating clients $\mathcal{C}$ in round $t$, the variance ratio for client $i$ is:

$$r_i^{(t)} = \frac{V_i}{\text{Med}_{j \in \mathcal{C}}(V_j)}$$

where $\text{Med}(\cdot)$ denotes the median function, chosen for its robustness to outliers.

**Detection Rule 4.1 (Variance-based Anomaly Detection).** Client $i$ is classified as suspicious if:

$$r_i^{(t)} < \tau_{var}$$

where $\tau_{var} \in (0, 1)$ is a dataset-specific threshold.

### 4.2.4 Adaptive Threshold Learning

To accommodate varying data heterogeneity levels across different federated learning scenarios, we propose an adaptive threshold learning mechanism.

**Algorithm 4.1: Adaptive Variance Threshold Calibration**

```
Input: 
    G_warmup: Set of gradients from warmup phase (assumed benign)
    p: Percentile parameter (default: 10)
    min_samples: Minimum samples for calibration (default: 50)
    
Output:
    τ_learned: Calibrated variance threshold

Procedure:
1.  V_history ← ∅
2.  for each gradient g in G_warmup do
3.      V ← Var(g)
4.      V_history ← V_history ∪ {V}
5.  end for
6.  
7.  if |V_history| ≥ min_samples then
8.      τ_learned ← Percentile(V_history, p)
9.      is_calibrated ← True
10. else
11.     τ_learned ← τ_default  // Use dataset-specific default
12.     is_calibrated ← False
13. end if
14. 
15. return τ_learned
```

**Remark 4.1.** The percentile parameter $p = 10$ is chosen to capture the lower tail of the benign variance distribution, as ALIE-transformed gradients exhibit systematically lower variance than benign gradients.

### 4.2.5 Dataset-Specific Parameterization

Empirical analysis across multiple datasets yields the following recommended thresholds:

| Dataset | $\tau_{var}$ | Rationale |
|---------|--------------|-----------|
| MNIST | 0.50 | Simple image structure, low natural variance |
| CIFAR-10 | 0.60 | Complex RGB images, moderate variance |
| FEMNIST | 0.40 | High heterogeneity by design, larger variance spread |

### 4.2.6 Complexity Analysis

**Proposition 4.1 (Computational Complexity).** *The variance-based detection mechanism has:*
- *Time complexity: $O(n \cdot d)$ per round, where $n$ is the number of clients and $d$ is the gradient dimension*
- *Space complexity: $O(n)$ for storing variance values*

*Proof.* Computing the variance of each gradient requires $O(d)$ operations. For $n$ clients, the total time is $O(n \cdot d)$. Storage requires only one scalar value per client. $\square$

---

## 4.3 Cross-round Consistency Detection

### 4.3.1 Motivation

While variance-based detection exploits instantaneous statistical properties, it may produce false negatives when attackers adaptively adjust their strategies. We complement this approach with **temporal analysis** that examines behavioral patterns across multiple training rounds.

**Key Observation:** Persistent attackers exhibit consistent behavioral patterns because:
1. Attack algorithms apply deterministic transformations
2. Poisoning objectives require sustained directional bias
3. Attacker's local data distribution remains constant

In contrast, benign clients exhibit natural variation due to:
1. Stochastic gradient descent randomness
2. Mini-batch sampling variation
3. Learning rate scheduling effects

### 4.3.2 Memory-Efficient Client Profiling

**Problem Statement:** Naive cross-round analysis requires storing complete gradient histories:

$$\text{Memory}_{naive} = n \times d \times w \times \text{sizeof(float32)}$$

For typical federated learning scenarios ($n = 40$, $d = 11 \times 10^6$, $w = 5$):

$$\text{Memory}_{naive} = 40 \times 11 \times 10^6 \times 5 \times 4 = 8.8 \text{ GB}$$

This is prohibitively expensive for edge deployment.

**Solution:** We propose storing only **sufficient statistics** that capture behavioral patterns:

**Definition 4.3 (Client Drift Profile).** For client $i$, the drift profile $\mathcal{P}_i$ is a tuple:

$$\mathcal{P}_i = (n_i, v_i, \rho_i, \mathbf{M}_i, \mathbf{D}_i, s_i, k_i, t_i)$$

where:
- $n_i \in \mathbb{R}^+$: Last observed gradient norm
- $v_i \in \mathbb{R}^+$: Last observed gradient variance  
- $\rho_i \in [0,1]$: Last observed positive element ratio
- $\mathbf{M}_i \in \mathbb{R}^w$: Rolling window of magnitude drifts
- $\mathbf{D}_i \in \{-1, 0, 1\}^w$: Rolling window of direction drifts
- $s_i \in [0,1]$: Consistency score
- $k_i \in \mathbb{N}$: Number of rounds tracked
- $t_i \in \mathbb{N}$: Last active round

**Proposition 4.2 (Memory Efficiency).** *The drift profile requires approximately 130 bytes per client.*

$$\text{Memory}_{optimized} = n \times 130 \text{ bytes}$$

For $n = 40$ clients: $\text{Memory}_{optimized} = 5.2$ KB.

**Memory Reduction Factor:**

$$\frac{\text{Memory}_{naive}}{\text{Memory}_{optimized}} = \frac{8.8 \times 10^9}{5.2 \times 10^3} \approx 1.7 \times 10^6$$

### 4.3.3 Drift Metrics

**Definition 4.4 (Magnitude Drift).** The magnitude drift for client $i$ between rounds $t-1$ and $t$ is:

$$\Delta_m^{(t)} = \frac{|n_i^{(t)} - n_i^{(t-1)}|}{n_i^{(t-1)} + \epsilon}$$

where $\epsilon = 10^{-10}$ prevents division by zero.

**Definition 4.5 (Direction Drift).** The direction drift is computed from the change in positive element ratio:

$$\Delta_d^{(t)} = \text{sign}\left(\rho_i^{(t)} - \rho_i^{(t-1)}\right)$$

where $\rho_i^{(t)} = \frac{1}{d}|\{j : g_{i,j}^{(t)} > 0\}|$ is the fraction of positive gradient elements.

**Remark 4.2.** The direction drift captures the tendency of gradient sign patterns to shift consistently in one direction, which is characteristic of persistent poisoning attacks.

### 4.3.4 Consistency Score Computation

**Definition 4.6 (Consistency Score).** The consistency score $s_i$ quantifies the regularity of drift patterns:

$$s_i = \alpha \cdot S_{dir} + (1 - \alpha) \cdot S_{mag}$$

where $\alpha = 0.6$ weights direction consistency higher than magnitude consistency.

**Direction Consistency Score:**

$$S_{dir} = \begin{cases}
1.0 & \text{if } \forall j, k \in \mathcal{N}: D_i[j] \cdot D_i[k] \geq 0 \\
0.3 & \text{otherwise}
\end{cases}$$

where $\mathcal{N} = \{j : D_i[j] \neq 0\}$ is the set of indices with non-zero direction drift.

**Magnitude Consistency Score:**

$$S_{mag} = 1 - \min\left(\beta \cdot \text{std}(\mathbf{M}_i), 1\right)$$

where $\beta = 5$ is a scaling factor and $\text{std}(\cdot)$ denotes standard deviation.

**Interpretation:**
- $S_{dir} = 1.0$: All direction drifts have the same sign (highly suspicious)
- $S_{mag} \approx 1.0$: Magnitude drifts have low variance (highly consistent)
- High $s_i$: Client exhibits persistent, consistent behavior (potential attacker)

### 4.3.5 Detection Algorithm

**Algorithm 4.2: Cross-round Consistency Detection**

```
Input:
    g_i: Gradient from client i
    i: Client identifier
    t: Current round
    τ_drift: Consistency threshold (default: 0.75)
    w: Window size (default: 5)
    
Output:
    is_suspicious: Boolean detection result
    
Global State:
    Profiles: Dictionary mapping client_id → ClientDriftProfile

Procedure:
1.  // Extract current round statistics
2.  n_curr ← ||g_i||_2
3.  v_curr ← Var(g_i)
4.  ρ_curr ← mean(g_i > 0)
5.  
6.  // Initialize profile if new client
7.  if i ∉ Profiles then
8.      Profiles[i] ← new ClientDriftProfile()
9.  end if
10. 
11. P ← Profiles[i]
12. 
13. // Compute drifts if not first observation
14. if P.k > 0 then
15.     Δ_m ← |n_curr - P.n| / (P.n + ε)
16.     Δ_d ← sign(ρ_curr - P.ρ)
17.     
18.     // Update rolling windows
19.     P.M.append(Δ_m)
20.     P.D.append(Δ_d)
21.     
22.     // Compute consistency score if sufficient history
23.     if |P.M| ≥ 3 then
24.         S_dir ← ComputeDirectionConsistency(P.D)
25.         S_mag ← ComputeMagnitudeConsistency(P.M)
26.         P.s ← 0.6 × S_dir + 0.4 × S_mag
27.     end if
28. end if
29. 
30. // Update profile state
31. P.n ← n_curr
32. P.v ← v_curr
33. P.ρ ← ρ_curr
34. P.k ← P.k + 1
35. P.t ← t
36. 
37. // Detection decision
38. if P.k ≥ 3 and P.s > τ_drift then
39.     return True
40. else
41.     return False
42. end if
```

### 4.3.6 Stale Profile Cleanup

To prevent memory growth in long-running deployments, we implement periodic cleanup of inactive client profiles:

**Algorithm 4.3: Stale Profile Cleanup**

```
Input:
    t: Current round
    τ_stale: Staleness threshold (default: 20 rounds)
    
Procedure:
1.  for each (client_id, profile) in Profiles do
2.      if t - profile.t > τ_stale then
3.          delete Profiles[client_id]
4.      end if
5.  end for
```

### 4.3.7 Complexity Analysis

**Proposition 4.3 (Computational Complexity).** *The cross-round consistency detection has:*
- *Time complexity: $O(n \cdot d)$ per round for gradient statistics, $O(n \cdot w)$ for consistency computation*
- *Space complexity: $O(n \cdot w)$ for storing drift histories*

---

## 4.4 Combined Detection: Layer 1.5

### 4.4.1 Integration Strategy

Layer 1.5 combines both detection mechanisms using a **union strategy**:

**Definition 4.7 (Combined Suspicious Set).** The set of suspicious clients identified by Layer 1.5 is:

$$\mathcal{S}_{L1.5} = \mathcal{S}_{var} \cup \mathcal{S}_{drift}$$

where:
- $\mathcal{S}_{var} = \{i : r_i < \tau_{var}\}$ (variance-flagged clients)
- $\mathcal{S}_{drift} = \{i : s_i > \tau_{drift} \land k_i \geq 3\}$ (consistency-flagged clients)

**Rationale:** The union strategy maximizes recall, accepting higher false positive rates in exchange for catching more sophisticated attackers. This is appropriate because:
1. Stealthy attacks cause significant model degradation if undetected
2. False positives are mitigated by the reputation system over time
3. The two detectors have complementary strengths

### 4.4.2 Status Escalation Mechanism

A critical design decision is the **escalation of detected clients to REJECTED status**, bypassing the Layer 2 rescue mechanism.

**Problem:** Layer 2 implements a rescue mechanism that accepts FLAGGED clients if their cosine similarity exceeds a threshold (typically 0.7). ALIE attackers consistently achieve cosine similarity > 0.99, resulting in false rescues:

$$\text{cosine}(\mathbf{g}_{ALIE}, \mathbf{g}_{median}) > 0.99 > \tau_{rescue} = 0.7$$

**Solution:** Layer 1.5 escalates suspicious clients directly to REJECTED status:

**Algorithm 4.4: Layer 1.5 Status Integration**

```
Input:
    L1_status: Dictionary of Layer 1 detection results
    S_L1.5: Set of clients flagged by Layer 1.5
    flag_as_rejected: Boolean (default: True)
    
Output:
    Updated L1_status

Procedure:
1.  target ← "REJECTED" if flag_as_rejected else "FLAGGED"
2.  
3.  for each client_id in S_L1.5 do
4.      current ← L1_status[client_id]
5.      
6.      if current = "ACCEPTED" then
7.          L1_status[client_id] ← target
8.      else if current = "FLAGGED" and flag_as_rejected then
9.          L1_status[client_id] ← "REJECTED"
10.     end if
11. end for
12. 
13. return L1_status
```

### 4.4.3 Detection Pipeline Flow

The complete detection flow with Layer 1.5 integration:

```
┌──────────────────────────────────────────────────────────────────────┐
│                     DETECTION PIPELINE                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Input: Client gradients {g_1, g_2, ..., g_n}                        │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────┐                         │
│  │         LAYER 1: DBSCAN + MAD           │                         │
│  │  • Magnitude Filter: ||g|| > θ_reject   │                         │
│  │  • DBSCAN Clustering: Outlier detection │                         │
│  │  • Output: {REJECTED, FLAGGED, ACCEPTED}│                         │
│  └─────────────────────────────────────────┘                         │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────┐                         │
│  │      LAYER 1.5: BEHAVIORAL ANALYSIS     │  ← Novel Contribution   │
│  │  • Variance Detection: r_i < τ_var      │                         │
│  │  • Consistency Detection: s_i > τ_drift │                         │
│  │  • Escalation: ACCEPTED → REJECTED      │                         │
│  └─────────────────────────────────────────┘                         │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────┐                         │
│  │     LAYER 2: DISTANCE + DIRECTION       │                         │
│  │  • Euclidean distance verification      │                         │
│  │  • Cosine similarity check              │                         │
│  │  • Rescue mechanism (for FLAGGED only)  │                         │
│  └─────────────────────────────────────────┘                         │
│                              │                                        │
│                              ▼                                        │
│  Output: Final client status {REJECTED, SUSPICIOUS, CLEAN}           │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4.5 Theoretical Guarantees

### 4.5.1 Detection Capability

**Theorem 4.2 (ALIE Detection Guarantee).** *Under the ALIE attack with parameter $z \geq 1$, the variance-based detector achieves non-zero detection rate when the variance reduction exceeds the threshold margin:*

$$\frac{\text{Var}(\mathbf{g}_{mal})}{\text{Var}(\mathbf{g}_{benign})} < \tau_{var}$$

**Theorem 4.3 (Persistent Attack Detection).** *For any attack that applies a deterministic transformation across rounds, the consistency detector will flag the attacker after $w$ rounds with probability approaching 1 as the number of observations increases.*

### 4.5.2 False Positive Analysis

**Proposition 4.4 (Expected False Positive Rate).** *Under the assumption that benign gradients follow independent distributions across rounds, the expected false positive rate for consistency detection is:*

$$\mathbb{E}[\text{FPR}] \leq \left(\frac{1}{2}\right)^{w-1}$$

*for a window size of $w$ rounds.*

---

## 4.6 Experimental Configuration

### 4.6.1 Hyperparameter Summary

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Variance threshold (CIFAR-10) | $\tau_{var}$ | 0.6 | Ratio threshold for variance detection |
| Consistency threshold | $\tau_{drift}$ | 0.75 | Score threshold for drift detection |
| Window size | $w$ | 5 | Number of rounds for drift history |
| Direction weight | $\alpha$ | 0.6 | Weight for direction consistency |
| Staleness threshold | $\tau_{stale}$ | 20 | Rounds before profile cleanup |
| Minimum observations | $k_{min}$ | 3 | Rounds before detection activation |

### 4.6.2 Configuration File

```toml
[defense.layer15]
# Enable/Disable Components
enable-variance = true
enable-drift = true

# Variance Detection Parameters
variance-ratio-mnist = 0.5
variance-ratio-cifar10 = 0.6
variance-ratio-femnist = 0.4

# Cross-round Consistency Parameters
drift-threshold = 0.75
drift-window = 5
drift-stale-rounds = 20
```

---

## 4.7 Summary

This chapter presented Layer 1.5: Behavioral Analysis, consisting of two complementary detection mechanisms:

1. **Variance-based Detection** exploits the statistical side effect of ALIE's gradient transformation, achieving detection through variance ratio analysis with adaptive threshold learning.

2. **Cross-round Consistency Detection** leverages temporal behavioral patterns, using memory-efficient client profiling (130 bytes per client) to identify persistent attackers through consistency scoring.

The key technical contributions include:
- Theoretical analysis of variance reduction under ALIE transformation
- Memory-efficient profiling achieving 1.7 million times reduction versus naive approaches  
- Status escalation mechanism to bypass Layer 2 false rescues
- Configurable parameters for different federated learning scenarios

Experimental results demonstrate detection rates of 60-85% against ALIE attacks that completely evade traditional detection methods, with acceptable false positive rates of 5-15%.