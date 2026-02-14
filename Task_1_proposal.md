# Cloud-Edge Simulation Benchmark for LLM Speculative Decoding - Mini Proposal

**Author:** [Ansuman Patra]  
**Project:** CNCF - KubeEdge / Ianvs: Cloud-Edge Simulation Benchmark for LLM Speculative Decoding  
**LFX 2026 Term - 1**

---

## 1. Problem Statement (Understanding)

### Why Speculative Decoding Speedups Are Not Guaranteed in Cloud-Edge Collaboration

Speculative decoding achieves acceleration by having a lightweight draft model generate K candidate tokens in parallel, which are then verified by a larger target model in a single forward pass. In ideal single-node deployments with low inter-model communication overhead, this can yield 2-3× speedups. However, cloud-edge collaboration introduces fundamental challenges that can eliminate or even reverse these gains:

**Network Latency Overhead:** Each speculation round requires bidirectional communication: edge→cloud (send K draft tokens + logits) and cloud→edge (return accepted tokens + rejection points). With typical RTTs of 20-100ms in edge deployments, this overhead can dominate the draft generation savings. If draft generation takes 50ms but network RTT is 80ms, the net gain diminishes significantly.

**Bandwidth Constraints:** Transmitting K draft tokens along with their probability distributions (vocabulary_size × K floats) can saturate limited edge uplinks (often 1-10 Mbps). A single speculation batch with K=10 and vocab_size=32,000 requires ~1.3MB of data transfer, taking 1+ seconds on a 10Mbps link. This creates a bottleneck that standard speculative decoding doesn't account for.

**Acceptance Rate Variability:** The core assumption of speculative decoding—that the draft model's predictions align well with the target model—breaks down when the edge draft model is aggressively compressed for resource constraints. Lower acceptance rates (α < 0.3) mean more wasted edge compute and network transfers, potentially making the overhead exceed the benefits of parallel verification.

**Heterogeneous Compute Asymmetry:** The speedup calculation assumes the draft model is significantly faster than the target model. However, edge devices (Jetson Nano, RPi) have vastly different compute profiles than cloud GPUs. If the draft model takes 200ms on edge hardware but verification takes 100ms on cloud GPU, the theoretical parallelism advantage is lost, especially when compounded by network delays.

**Jitter and Queuing:** Real-world networks exhibit variable latency (jitter), and cloud services batch requests for throughput. These introduce unpredictable delays that disrupt the tight timing assumptions of speculative decoding, causing the edge to either wait idle or generate excessive draft tokens that get rejected.

### How Network Latency/Bandwidth and Heterogeneous Compute Change End-to-End Gains

The end-to-end latency in cloud-edge speculative decoding becomes:

```
L_total = N_rounds × (T_edge_draft + T_transfer_up + T_queue_cloud + T_cloud_verify + T_transfer_down)
```

where `N_rounds = T_total_tokens / α × K` (inversely proportional to acceptance rate).

**Critical Break-Even Analysis:**
- **Standard autoregressive cloud decoding:** `T_auto = T_total_tokens × (T_cloud_forward + 2×RTT)`
- **Speculative cloud-edge:** `T_spec = N_rounds × (T_edge_draft + 2×RTT + T_cloud_verify + bandwidth_overhead)`

Speculative decoding only wins when:
```
T_spec < T_auto
⟹ α > threshold where threshold = f(RTT, bandwidth, K, compute_ratio)
```

For example, with RTT=50ms, K=5, α=0.4, edge_draft=100ms, cloud_verify=80ms:
- Autoregressive: 50 tokens × (100ms + 100ms) = 10,000ms
- Speculative: 25 rounds × (100ms + 100ms + 80ms + 20ms_transfer) = 7,500ms (25% speedup)

But with RTT=100ms and α=0.3:
- Speculative: 33 rounds × (100ms + 200ms + 80ms + 40ms_transfer) = 13,860ms (39% slowdown)

The gains are highly non-linear and dependent on the interplay of all factors. Moreover, heterogeneous compute can create **idle time asymmetries**: while the cloud verifies, the edge sits idle (or generates too many drafts that get rejected). Conversely, while the edge drafts, the cloud may be serving other requests, causing queuing delays.

---

## 2. Benchmark Scope (What to Evaluate)

To comprehensively evaluate cloud-edge speculative decoding, the benchmark must cover the following key variables:

### 2.1 Network Characteristics
1. **Round-Trip Time (RTT):** 10ms (LAN), 30ms (metro-edge), 50ms (typical edge), 100ms (remote edge), 200ms (satellite/3G)
2. **Bandwidth:** 100Mbps (fiber), 10Mbps (typical edge), 1Mbps (constrained IoT), with separate up/down configurations
3. **Jitter:** Standard deviation of RTT (±5ms, ±20ms, ±50ms) to simulate realistic network variability
4. **Packet Loss Rate:** 0%, 1%, 5% to test robustness under degraded network conditions

### 2.2 Compute Heterogeneity
1. **Draft Model Size / Edge Compute:** Tiny (68M on Jetson Nano), Small (1B on RPi 5), Medium (7B on edge GPU)
2. **Target Model Size / Cloud Compute:** Standard (7B on A100), Large (13B on A100), Extra-Large (70B on multi-GPU)
3. **Compute Ratio:** Time ratio between edge draft generation and cloud verification (0.5×, 1×, 2×, 5×)

### 2.3 Speculation Parameters
1. **Draft Size (K):** Number of speculative tokens per round (1, 3, 5, 7, 10, 15, 20)
2. **Temperature:** Sampling temperature for draft model (0.0=greedy, 0.7, 1.0) affecting acceptance rate
3. **Acceptance Threshold:** Probability threshold for accepting draft tokens (affects α and quality trade-offs)

### 2.4 Workload Characteristics
1. **Prompt Length:** Short (10 tokens), Medium (100 tokens), Long (512 tokens), Extra-Long (2048 tokens)
2. **Generation Length:** 50, 100, 200, 500, 1000 tokens
3. **Task Complexity:** Simple QA (high α), code generation (medium α), creative writing (low α)
4. **Concurrency:** Single user, 5 concurrent requests, 10 concurrent requests (to test queuing effects)

### 2.5 System Configuration
1. **Batching Strategy:** Client-side batching of speculative rounds vs. server-side batching of verifications
2. **Caching:** KV-cache management between edge and cloud (local-only, shared, hybrid)
3. **Prefetching:** Whether edge pre-generates next draft batch while waiting for verification

---

## 3. Metrics & Methodology (How to Measure)

### 3.1 Primary Metrics

#### Time-to-First-Token (TTFT)
**Definition:** Latency from query submission to first token generated back to the client.

**Timing Boundaries:**
- **Start:** Timestamp when user query is received at edge node
- **End:** Timestamp when first verified token arrives back at edge node
- **Includes:** Initial prompt encoding, first draft generation, network transfer to cloud, first verification, response transfer back

**Why It Matters:** Critical for perceived responsiveness in interactive applications like chatbots. Cloud-edge speculative decoding may actually increase TTFT due to initial round-trip overhead compared to direct cloud inference.

**Measurement Method:**
```python
t_start = time.perf_counter()
first_token = await edge_to_cloud_speculation(prompt)
t_first_token = time.perf_counter() - t_start
```

#### Tokens Per Second (tokens/s)
**Definition:** Throughput of token generation averaged over the entire generation process.

**Calculation:**
```
tokens/s = total_output_tokens / (t_end - t_start)
```

**Timing Boundaries:**
- **Start:** Same as TTFT start
- **End:** Timestamp when final token is generated and delivered to client
- **Excludes:** User think time, prompt tokenization preprocessing
- **Includes:** All speculation rounds, network transfers, idle/queuing time

**Percentile Reporting:** Report p50, p95, p99 to capture variance across different prompt/generation lengths and network conditions.

#### End-to-End Latency (E2E)
**Definition:** Total time from query submission to complete response delivery.

**Decomposition:**
```
E2E = TTFT + (total_tokens - 1) × inter_token_latency_avg
```

**Sub-Components Logged:**
- Edge draft time per round
- Network upload time per round
- Cloud queue wait time
- Cloud verification time per round
- Network download time per round
- Edge processing/acceptance time

### 3.2 Secondary Metrics

#### Acceptance Rate (α)
**Definition:** Fraction of draft tokens accepted by target model per speculation round.

**Calculation:**
```
α = accepted_tokens / total_drafted_tokens
```

**Importance:** Core indicator of draft-target alignment. α < 0.4 typically means speculative decoding is ineffective.

#### Speculation Efficiency
**Definition:** Ratio of useful work to total work done.

**Formula:**
```
efficiency = accepted_tokens / (edge_drafts + cloud_forward_passes)
```

Higher efficiency indicates better resource utilization.

#### Network Utilization
**Definition:** Fraction of time spent waiting for network vs. compute.

**Calculation:**
```
network_overhead = sum(t_network_transfers) / E2E
```

Target: < 30% for speculative decoding to be viable.

#### Cost Metrics
**Definition:** Relative API cost compared to standard cloud inference.

**Formula:**
```
cost_ratio = (edge_compute_cost + cloud_api_calls × cloud_cost) / baseline_cloud_cost
```

Edge-cloud speculative should ideally reduce cost by shifting compute to cheaper edge while maintaining quality.

### 3.3 Timing Instrumentation Strategy

**Warmup Handling:** 
- Run 3 warmup iterations before benchmarking to eliminate cold-start effects
- Clear KV-caches between benchmark runs to ensure reproducibility
- Report both cold-start and warm-start metrics separately

**Network Wait Inclusion:**
- **Always include** network transfer time in all metrics
- Log separate network time as a sub-component for diagnosis
- Do NOT exclude network overhead as it's integral to cloud-edge performance

**Clock Synchronization:**
- Use `time.perf_counter()` for high-resolution timing
- For multi-node benchmarks, use NTP-synchronized clocks with <1ms drift
- Report timing from edge perspective (user's view) not cloud

**Statistical Validity:**
- Run each configuration 10+ times
- Report mean, std dev, p50, p95, p99
- Use fixed random seeds for draft sampling to enable reproducibility

---

## 4. High-Level Design (How to Land in Ianvs)

### 4.1 Ianvs Component Mapping

**Benchmark Structure:**
```
examples/cloud-edge-speculative-decoding/
├── testenv.yaml                    # Test environment configuration
├── benchmarking_job.yaml           # Job orchestration
├── testalgorithms/
│   ├── baseline_autoregressive/    # Baseline: Standard cloud-only decoding
│   ├── vanilla_speculative/        # Vanilla speculative decoding
│   └── adaptive_speculative/       # Our novel method (Task 1.6)
├── testcases/
│   ├── low_latency_scenario.yaml   # RTT=10ms, BW=100Mbps
│   ├── typical_edge_scenario.yaml  # RTT=50ms, BW=10Mbps
│   └── constrained_scenario.yaml   # RTT=100ms, BW=1Mbps
└── data/
    ├── prompts/
    │   ├── simple_qa.jsonl
    │   ├── code_generation.jsonl
    │   └── creative_writing.jsonl
    └── models/
        ├── draft_model/            # Vicuna-68M or TinyLlama-1B
        └── target_model/           # Llama-2-7B or Qwen-7B
```

**Paradigm Selection:** Custom `cloud_edge_speculation` paradigm extending Ianvs' paradigm base.

**Core Components:**

1. **TestEnvironmentManager:** 
   - Manage model versions, network simulator configs, dataset paths
   - Define metrics computation classes

2. **TestCaseController:**
   - Load scenario configs (RTT, bandwidth, K, etc.)
   - Instantiate edge/cloud process simulators
   - Orchestrate benchmark runs with parameter sweeps

3. **Simulation Controller:**
   - **EdgeSimulator:** Runs draft model in separate process, simulates edge hardware constraints
   - **CloudSimulator:** Runs target model, simulates batching and queuing
   - **NetworkSimulator:** Injects latency, bandwidth limits, jitter, packet loss using `tc` (Linux traffic control) or Python-based emulation

4. **Story Manager:**
   - Generate comparative reports (baseline vs speculative)
   - Visualizations: acceptance rate vs speedup, RTT vs efficiency, etc.
   - Leaderboard for different algorithm implementations

### 4.2 Reproducible Configuration Approach

**Determinism Guarantees:**

1. **Fixed Random Seeds:**
```yaml
random_seed: 42
torch_seed: 42
numpy_seed: 42
```

2. **Pinned Model Versions:**
```yaml
models:
  draft:
    name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    revision: "fe8a4ea1ffedaf415f4da2f062534de366a451e6"  # Git commit hash
  target:
    name: "meta-llama/Llama-2-7b-chat-hf"
    revision: "c1b0db933684edbfe29a06fa47eb19cc48025e93"
```

3. **Controlled Sampling:**
```yaml
generation_config:
  do_sample: false  # Use greedy decoding for determinism
  temperature: 0.0
  top_p: 1.0
  max_new_tokens: 100
```

4. **Network Determinism:**
```yaml
network_simulation:
  type: "deterministic"  # vs "stochastic"
  rtt: 50  # ms, fixed
  bandwidth: 10  # Mbps
  jitter: 0  # ms, disable for reproducibility
```

### 4.3 Draft Configuration Schema

```yaml
# benchmarking_config.yaml
benchmark_job:
  name: "cloud-edge-speculative-decoding-benchmark"
  workspace: "./workspace"
  
  test_environment:
    dataset:
      name: "llm_prompts_mix"
      path: "./data/prompts/mixed_tasks.jsonl"
      samples: 100
    
    models:
      draft_model:
        name: "TinyLlama-1.1B"
        path: "./models/draft/"
        device: "cpu"  # Simulate edge constraints
      target_model:
        name: "Llama-2-7B"
        path: "./models/target/"
        device: "cuda"  # Simulate cloud GPU
    
    metrics:
      - name: "ttft"
        class: "TimeToFirstTokenMetric"
      - name: "tokens_per_second"
        class: "ThroughputMetric"
        percentiles: [50, 95, 99]
      - name: "acceptance_rate"
        class: "AcceptanceRateMetric"
      - name: "e2e_latency"
        class: "EndToEndLatencyMetric"
  
  test_scenarios:
    - name: "low_latency_optimal"
      network:
        rtt_ms: 10
        bandwidth_mbps: 100
        jitter_ms: 2
      speculation:
        draft_size_k: 5
        max_rounds: 50
      expected_speedup: 2.0  # Hypothesis for validation
    
    - name: "typical_edge"
      network:
        rtt_ms: 50
        bandwidth_mbps: 10
        jitter_ms: 10
      speculation:
        draft_size_k: 5
        max_rounds: 50
      expected_speedup: 1.3
    
    - name: "constrained_edge"
      network:
        rtt_ms: 100
        bandwidth_mbps: 1
        jitter_ms: 20
      speculation:
        draft_size_k: 3  # Reduce K for bandwidth
      expected_speedup: 0.8  # May be slower!
  
  algorithms:
    - name: "baseline_autoregressive"
      class: "BaselineAutoregressive"
      params:
        deployment: "cloud_only"
    
    - name: "vanilla_speculative"
      class: "VanillaSpeculativeDecoding"
      params:
        draft_size_k: 5
        temperature: 0.0
    
    - name: "adaptive_speculative"
      class: "AdaptiveNetworkAwareSpeculative"
      params:
        initial_k: 5
        adaptive_k: true
        prefetch: true
  
  output:
    report_format: ["json", "html", "csv"]
    visualizations:
      - "speedup_vs_rtt"
      - "acceptance_vs_k"
      - "efficiency_breakdown"
```

---

## 5. Milestones (9-12 Weeks Plan)

### **Milestone 1: Foundation & Baseline (Weeks 1-3)**

**Deliverables:**
1. Ianvs paradigm template for cloud-edge LLM inference
2. Network simulation module with configurable RTT/bandwidth/jitter
3. Baseline autoregressive cloud-only inference implementation
4. Initial test environment with 2 model pairs (TinyLlama+Llama-7B, Vicuna-68M+Qwen-7B)
5. Basic metrics collection (TTFT, tokens/s, E2E latency)

**Acceptance Criteria:**
- Baseline runs successfully on at least 3 RTT settings (10ms, 50ms, 100ms)
- Network simulator verified to inject target latency ±5% accuracy
- Metrics logged to JSON with proper timestamps
- Can generate 100 tokens with deterministic output (fixed seed)

**Validation:**
```bash
# Should complete in <5 seconds on localhost
python run_baseline.py --rtt 10 --bandwidth 100 --tokens 100 --verify-determinism
```

---

### **Milestone 2: Vanilla Speculative Decoding (Weeks 4-6)**

**Deliverables:**
1. Vanilla speculative decoding algorithm implementation
2. Edge/cloud process isolation (separate Python processes or Docker containers)
3. Inter-process communication with serialized token transfers
4. Acceptance rate calculation and logging
5. Comparative benchmarking: baseline vs vanilla speculative across 5 scenarios
6. Draft report with initial speedup findings

**Acceptance Criteria:**
- Vanilla speculative achieves >1.5× speedup in low-latency (RTT=10ms) scenario
- Acceptance rate logged per-round with mean >0.5 for aligned models
- Benchmark completes 500-token generation with K=5 in all 5 test scenarios
- Report includes speedup vs RTT plot showing expected inverse relationship
- Edge and cloud processes run independently (verified by separate PIDs)

**Validation:**
```bash
# Should show speedup > 1.5x for RTT=10ms, acceptance > 0.5
python run_speculative.py --scenario low_latency --k 5 --report speedup_analysis.json
```

---

### **Milestone 3: Adaptive Algorithm & Comprehensive Evaluation (Weeks 7-10)**

**Deliverables:**
1. Adaptive network-aware speculative decoding (our novel algorithm from Section 6)
2. Prefetching and pipelined draft generation
3. Comprehensive parameter sweep: K ∈ {1,3,5,7,10}, RTT ∈ {10,30,50,100,200}, BW ∈ {1,10,100}
4. Statistical analysis with confidence intervals (10 runs per config)
5. Break-even analysis: regions where speculative wins/loses
6. Detailed profiling: time breakdown per component (draft, network, verify, queue)

**Acceptance Criteria:**
- Adaptive algorithm outperforms vanilla by >15% in high-latency scenarios (RTT>80ms)
- Benchmark completes 50+ configurations in <6 hours on test hardware
- Statistical significance: p95 confidence intervals have <10% relative width
- Report identifies optimal K for each RTT/BW combination
- Leaderboard generated with top 3 algorithms ranked by cost-adjusted efficiency

**Validation:**
```bash
# Comprehensive sweep
python run_benchmark_suite.py --algorithms all --scenarios comprehensive --output results/
# Should generate: leaderboard.html, efficiency_map.png, break_even_chart.png
```

---

### **Milestone 4: Finalization & Documentation (Weeks 11-12)**

**Deliverables:**
1. Complete Ianvs example with README and quick-start guide
2. Documentation: architecture diagrams, config schema reference, troubleshooting
3. Test suite: unit tests for metrics, integration tests for scenarios
4. CI/CD integration: automated benchmarking on commit
5. Final report: comprehensive benchmark results, analysis, recommendations
6. Community artifacts: blog post, presentation slides for KubeEdge SIG AI

**Acceptance Criteria:**
- A new user can run the full benchmark suite following the README in <30 minutes
- All tests pass in CI environment (GitHub Actions)
- Documentation covers 100% of config parameters with examples
- Final report published as Ianvs proposal with peer review
- At least 1 community presentation delivered (virtual or in-person)

**Validation:**
```bash
# New user quick-start
git clone <repo>
cd examples/cloud-edge-speculative-decoding
./scripts/setup.sh  # Installs deps, downloads models
./scripts/run_demo.sh  # Runs 3-scenario demo in <5 minutes
pytest tests/  # All tests pass
```

---

## 6. Innovative Acceleration Idea (Required)

### **Method Name: Adaptive Network-Aware Speculative Decoding with Preemptive Pipelining (ANSD-PP)**

---

### 6.1 Mechanism (What)

**Core Innovation:** ANSD-PP dynamically adjusts speculation length (K) based on real-time network conditions and introduces pipelined preemptive drafting to hide network latency by overlapping edge computation with cloud verification.

#### **Key Changes Compared to Standard Speculative Decoding:**

**1. Dynamic Speculation Length (K) Adaptation:**

Standard speculative decoding uses a fixed K for all speculation rounds. ANSD-PP adjusts K per-round using a lightweight controller:

```python
def adaptive_k_selection(current_rtt, current_bw, recent_acceptance_rate):
    """
    Adjust K based on network conditions to minimize wasted edge compute.
    """
    # Base K on network roundtrip efficiency
    baseline_k = max(1, int((current_bw_mbps * 0.1) / (current_rtt_ms / 10)))
    
    # Penalize low acceptance rates (indicates poor draft-target alignment)
    if recent_acceptance_rate < 0.4:
        k = max(1, baseline_k - 2)  # Reduce speculation aggressiveness
    elif recent_acceptance_rate > 0.7:
        k = min(20, baseline_k + 2)  # Increase speculation
    else:
        k = baseline_k
    
    # Hard cap based on bandwidth: avoid saturating uplink
    max_k_by_bandwidth = int((current_bw_mbps * 125000) / (vocab_size * 4 * 100))  # bytes/s / bytes_per_token
    k = min(k, max_k_by_bandwidth)
    
    return k
```

**Rationale:** When RTT is high but bandwidth is low, generating many draft tokens wastes edge compute and saturates the network. When acceptance is low, edge should draft fewer tokens to minimize rejection overhead. Conversely, in low-latency high-bandwidth scenarios with high α, K should increase to maximize parallelism.

**2. Preemptive Pipelined Drafting:**

Standard speculative decoding has the edge sit idle while waiting for cloud verification. ANSD-PP introduces **early-start preemptive drafting**:

```
Standard Flow:
Edge: [Draft K tokens] → (wait for cloud) → [Draft K tokens] → (wait) → ...
Cloud:                   [Verify K tokens] → (wait for edge) → [Verify K tokens] → ...

ANSD-PP Flow:
Edge: [Draft K1] → [Prefetch Draft K2] → [Prefetch Draft K3] → ...
Cloud:             [Verify K1] → [Verify K2] → [Verify K3] → ...
      ^--- Edge sends K1       ^--- Edge sends K2 (before K1 result returns)
```

The edge maintains a **speculation buffer** of 2-3 draft batches:
- While waiting for verification of batch N, edge generates batch N+1 and N+2
- When batch N result arrives, edge immediately sends N+1 (if N was fully accepted) or adjusts N+1 based on rejection point
- This hides network latency behind edge computation

**Algorithm:**
```python
async def pipelined_speculation(prompt, max_rounds):
    draft_buffer = asyncio.Queue(maxsize=3)
    results_queue = asyncio.Queue()
    
    # Edge task: continuously draft
    async def edge_drafter():
        context = prompt
        for round in range(max_rounds):
            k = adaptive_k_selection(...)
            draft_tokens = await draft_model.generate(context, k)
            await draft_buffer.put(draft_tokens)
            context = update_context(context, draft_tokens)  # Speculative
    
    # Cloud task: verify drafts as they arrive
    async def cloud_verifier():
        while True:
            draft_tokens = await draft_buffer.get()
            accepted, rejection_point = await target_model.verify(draft_tokens)
            await results_queue.put((accepted, rejection_point))
    
    # Run both concurrently
    await asyncio.gather(edge_drafter(), cloud_verifier())
```

**3. Early Exit Signal Passing:**

Inspired by recent work on early exits in transformers, ANSD-PP adds **intermediate verification signals**:
- Cloud sends preliminary acceptance likelihood after processing first L layers (e.g., L=8 out of 32)
- If early signal predicts low acceptance (<30% likelihood), cloud sends abort signal
- Edge immediately stops wasting compute on speculative drafts for that branch

**Implementation:**
```python
class EarlyExitVerifier:
    def __init__(self, target_model, early_exit_layer=8):
        self.model = target_model
        self.early_exit_layer = early_exit_layer
    
    async def verify_with_early_exit(self, draft_tokens):
        # Process first 8 layers
        early_logits = self.model.forward_partial(draft_tokens, layers=range(8))
        early_acceptance_prob = compute_acceptance_likelihood(early_logits, draft_tokens)
        
        if early_acceptance_prob < 0.3:
            await send_abort_signal_to_edge()
            return [], 0  # Reject all, save compute
        
        # Continue full verification
        full_logits = self.model.forward(draft_tokens)
        accepted, rejection_point = standard_verification(full_logits, draft_tokens)
        return accepted, rejection_point
```

---

### 6.2 Why It Helps in Cloud-Edge (Why)

**Explicit Connection to Cloud-Edge Constraints:**

#### **Constraint 1: High Round-Trip Time (RTT)**

**Problem:** Standard speculative decoding assumes low communication overhead. With RTT=100ms, each speculation round incurs 100ms idle wait, destroying parallelism gains.

**ANSD-PP Solution:** 
- **Preemptive pipelining** overlaps edge draft generation with network transfer and cloud verification
- While waiting for cloud response (100ms), edge generates next 2-3 draft batches (3×50ms = 150ms)
- Effectively **hides 66% of network latency** behind computation
- Reduces effective RTT impact: from 100ms×N_rounds to ~33ms×N_rounds

**Quantitative Benefit:** 
- Baseline speculative with RTT=100ms: 20 rounds × 100ms idle = 2000ms wasted
- ANSD-PP: 20 rounds × 33ms idle = 660ms wasted → **67% reduction in idle time**

#### **Constraint 2: Limited Bandwidth**

**Problem:** Fixed K in standard speculative can saturate edge uplink. K=10 with vocab_size=32K requires 1.3MB per round. On 1Mbps link, that's 10.4 seconds per round—far longer than generation itself!

**ANSD-PP Solution:**
- **Adaptive K** scales down speculation length when bandwidth is constrained
- When BW=1Mbps, controller sets K=2 instead of 10, reducing transfer to 0.26MB (2.08s vs 10.4s)
- Avoids network saturation while maintaining some speculation benefit

**Quantitative Benefit:**
- Standard K=10 on 1Mbps: 10.4s transfer × 20 rounds = 208s total transfer time
- ANSD-PP K=2 on 1Mbps: 2.08s transfer × 20 rounds = 41.6s → **80% reduction**

#### **Constraint 3: Heterogeneous Compute (Edge CPU vs Cloud GPU)**

**Problem:** Standard speculative assumes draft model is 5-10× faster than target. But edge CPU (200ms/token) vs cloud GPU (50ms/token) reduces the draft advantage to 4×, barely justifying the overhead.

**ANSD-PP Solution:**
- **Adaptive K** recognizes slow edge compute and reduces speculation aggressiveness
- When edge_draft_time > 0.5 × cloud_verify_time, controller caps K at 3-5 to avoid wasting edge cycles
- **Early exit aborts** prevent edge from spending 200ms on drafts that will be rejected

**Quantitative Benefit:**
- Standard K=10 with slow edge: 10 × 200ms draft = 2000ms edge compute per round
- If acceptance is 0.4, 60% of drafts are wasted → 1200ms wasted compute
- ANSD-PP K=5 + early abort: 5 × 200ms = 1000ms, with abort saving 500ms on rejected branches → **50% compute savings**

#### **Constraint 4: Network Jitter and Queuing**

**Problem:** Speculative decoding assumes predictable timing. But jitter (±30ms) and cloud queuing (variable 10-200ms) create timing uncertainty, causing edge to either over-draft (wasted) or under-draft (missed parallelism).

**ANSD-PP Solution:**
- **Dynamic K updates every N rounds** (e.g., every 5 rounds) based on observed RTT moving average
- If RTT spikes due to congestion, K is reduced immediately to avoid backlog
- **Prefetch buffer** absorbs jitter variance: even if verification is delayed, edge has speculative drafts ready

**Quantitative Benefit:**
- Standard speculative with ±30ms jitter: some rounds wait 130ms (vs expected 100ms), breaking parallelism
- ANSD-PP: buffer absorbs timing variance, maintains 90%+ utilization even with 30% jitter

---

### 6.3 Trade-offs / Risks

**1. Increased Edge Compute Overhead (Preemptive Over-Drafting)**

**Risk:** Pipelined prefetching may generate draft batches that get invalidated if earlier batches are fully rejected, wasting edge compute.

**Mitigation:** 
- Implement **rejection-aware prefetching**: only prefetch if recent acceptance rate > 0.5
- Use **speculative context snapshots**: if batch N is rejected at token 3, rollback prefetch context to match rejection point

**Expected Impact:** 10-20% additional edge compute in worst case, but 2× speedup in high-latency scenarios justifies cost.

**2. Lower Acceptance Rate Due to Divergence (Pipelined Context Mismatch)**

**Risk:** Prefetching based on speculative context (before verification) may cause draft tokens to diverge further from target model's true trajectory, reducing α.

**Quantitative Trade-Off:**
- Baseline α = 0.6 (no prefetch)
- ANSD-PP α = 0.5 (with prefetch)
- But latency reduction from pipelining (67% idle time saved) outweighs 17% drop in α
- Net speedup: 0.5/0.6 × 3.0× = 2.5× (still 2.5× faster despite lower α)

**Mitigation:**
- **Hybrid mode**: only prefetch 1 batch ahead (not 2-3) to limit divergence
- **Confidence thresholding**: only prefetch if draft model's confidence (max logit prob) > 0.7

**3. Implementation Complexity (Async Coordination, Edge Cases)**

**Risk:** Pipelined async architecture with buffering, early exits, and adaptive K introduces complex state management and potential race conditions.

**Mitigation:**
- Use well-tested async frameworks (asyncio, Ray)
- Comprehensive testing with edge case scenarios: all rejections, network timeouts, out-of-order arrivals
- Fallback to vanilla speculative if ANSD-PP coordination fails

**Expected Impact:** 2-3 weeks additional development time, but manageable with modular design.

**4. Determinism Loss (Adaptive K, Prefetching)**

**Risk:** Dynamic K and preemptive drafts may produce non-deterministic outputs if not carefully controlled, breaking reproducibility.

**Mitigation:**
- **Deterministic adaptive controller**: K is a pure function of network metrics (no randomness)
- **Seeded draft generation**: use fixed seeds even in prefetch branches
- **Replay capability**: log all K decisions and network measurements for exact reproduction

**5. Marginal Gains in Near-Optimal Scenarios**

**Risk:** In low-latency, high-bandwidth scenarios (RTT=10ms, BW=100Mbps), standard speculative already achieves 2.5× speedup. ANSD-PP's adaptations add complexity for minimal benefit.

**Expected Impact:** ANSD-PP provides <10% additional speedup in optimal scenarios, but that's acceptable—its value is in **preventing slowdowns** in suboptimal scenarios where standard speculative fails.

**Design Philosophy:** ANSD-PP is a **robust, adaptive approach** that gracefully degrades in poor conditions rather than assuming ideal networks.

---

### 6.4 Evaluation Plan (How to Verify)

#### **Baselines**

1. **Baseline Autoregressive (cloud-only):** Standard LLM inference with all generation on cloud GPU, edge only sends prompt. Establishes worst-case latency.

2. **Vanilla Speculative Decoding:** Fixed K=5, no adaptations, no prefetching. State-of-the-art baseline for cloud-edge collaboration.

3. **ANSD-PP (Our Method):** Full adaptive K + preemptive pipelining + early exits.

#### **Experimental Variables (2+ Variables per Hypothesis)**

**Primary Independent Variables:**
1. **RTT:** 10ms, 30ms, 50ms, 80ms, 100ms, 150ms, 200ms (7 levels)
2. **Bandwidth:** 100Mbps, 10Mbps, 5Mbps, 1Mbps (4 levels)
3. **Draft Size K (for vanilla):** 1, 3, 5, 7, 10 (5 levels) [ANSD-PP adapts K automatically]
4. **Task Complexity (affects α):** Simple QA (α≈0.7), code gen (α≈0.5), creative (α≈0.3)
5. **Concurrency:** 1, 5, 10 users (tests queuing effects)

**Total Configurations:** 7 RTT × 4 BW × 5 K × 3 tasks = 420 configs per baseline × 3 baselines = 1260 experiments

**Dependent Variables:**
- **Primary:** TTFT (ms), E2E latency (ms), tokens/s
- **Secondary:** Acceptance rate (α), edge compute time, network overhead %, cost ratio

#### **Metrics (with p95)**

All metrics reported as: **mean ± std dev, p50, p95, p99** across 10 runs per config.

**Core Metrics:**
1. **TTFT (p95):** 95th percentile time-to-first-token. Critical for user experience variability.
2. **Tokens/s (p50):** Median throughput. Less affected by outliers than mean.
3. **E2E Latency (p95):** 95th percentile end-to-end latency. Captures worst-case scenarios.
4. **Speedup Ratio:** `latency_baseline / latency_method`. Must be >1.1 to be meaningful.
5. **Efficiency Score:** `(speedup × α) / (1 + compute_overhead)`. Balances speed vs waste.

#### **Hypothesis & Expected Break-Even Regions**

**Hypothesis 1 (Wins):** ANSD-PP outperforms vanilla speculative in high-latency scenarios.

**Condition:** RTT ≥ 80ms OR Bandwidth ≤ 5Mbps

**Expected Results:**
- RTT=100ms, BW=10Mbps: ANSD-PP achieves 1.8× speedup vs 1.2× for vanilla (50% improvement)
- RTT=150ms, BW=1Mbps: ANSD-PP achieves 1.5× speedup vs 0.9× for vanilla (vanilla slower than baseline!)

**Break-Even Point:** RTT ≈ 60ms with BW=10Mbps (both methods tied at ~1.4× speedup)

**Hypothesis 2 (Marginal):** ANSD-PP provides minimal gain in optimal conditions.

**Condition:** RTT ≤ 30ms AND Bandwidth ≥ 100Mbps

**Expected Results:**
- RTT=10ms, BW=100Mbps: ANSD-PP 2.6× vs vanilla 2.5× (4% improvement, within noise)
- Justification: Network is not the bottleneck, adaptive K converges to vanilla's fixed K

**Hypothesis 3 (Loses):** ANSD-PP underperforms in extreme low-acceptance scenarios.

**Condition:** α < 0.25 (e.g., creative writing with mismatched draft model)

**Expected Results:**
- α=0.2: ANSD-PP 0.8× vs vanilla 0.7× vs baseline 1.0× (all methods slower due to poor alignment)
- Mitigation: Early abort prevents some wasted compute, but fundamental mismatch dominates

**Break-Even Acceptance:** α ≈ 0.35 (below this, speculative decoding in general becomes counterproductive)

**Hypothesis 4 (Scales):** ANSD-PP maintains efficiency under concurrency.

**Condition:** 5-10 concurrent users (tests queuing/batching)

**Expected Results:**
- Concurrency=10, RTT=50ms: ANSD-PP maintains 1.6× speedup vs vanilla's 1.3× (queuing tolerance)
- Reason: Preemptive pipelining keeps edge busy, reducing impact of cloud queue delays

#### **Statistical Validation**

**Sample Size:** 10 runs per config (sufficient for p95 estimation with 90% confidence)

**Significance Testing:** 
- Paired t-test between vanilla and ANSD-PP for each config
- Report p-values; flag speedups with p<0.05 as statistically significant

**Reproducibility:**
- Publish full config files, random seeds, model checkpoints
- Provide Docker container for exact environment replication

---

### 6.5 Ianvs Integration (Minimal Requirements)

#### **New Config Fields Required**

```yaml
algorithm:
  name: "ANSD-PP"
  class: "AdaptiveNetworkAwareSpeculativeDecoding"
  params:
    # Core parameters
    initial_k: 5
    adaptive_k_enabled: true
    k_adaptation_interval: 5  # Re-compute K every 5 rounds
    
    # Preemptive pipelining
    prefetch_enabled: true
    prefetch_buffer_size: 3  # Max drafts in pipeline
    prefetch_rejection_threshold: 0.5  # Only prefetch if α > 0.5
    
    # Early exit
    early_exit_enabled: true
    early_exit_layer: 8
    early_exit_threshold: 0.3  # Abort if acceptance_prob < 0.3
    
    # Network adaptation
    rtt_estimation_window: 10  # Moving average over last 10 rounds
    bandwidth_estimation_enabled: true
```

#### **Logging Fields for Benchmarking**

```python
per_round_logs = {
    "round_id": int,
    "timestamp_start": float,
    "timestamp_end": float,
    
    # Speculation parameters (what ANSD-PP decided)
    "draft_size_k": int,  # Adaptive K for this round
    "prefetch_count": int,  # How many batches were prefetched
    
    # Network measurements
    "measured_rtt_ms": float,
    "measured_bandwidth_mbps": float,
    "network_jitter_ms": float,
    
    # Timing breakdown
    "edge_draft_time_ms": float,
    "network_upload_time_ms": float,
    "cloud_queue_wait_ms": float,
    "cloud_verify_time_ms": float,
    "network_download_time_ms": float,
    
    # Acceptance metrics
    "drafted_tokens": int,
    "accepted_tokens": int,
    "acceptance_rate": float,
    "rejection_point": int,  # Index of first rejected token
    
    # Early exit (if enabled)
    "early_exit_triggered": bool,
    "early_acceptance_prob": float,
    
    # Efficiency
    "edge_compute_wasted_ms": float,  # Time on rejected drafts
    "network_utilization_pct": float
}
```

#### **Benchmark Report Extensions**

**New visualizations:**
1. `k_adaptation_timeline.png`: Plot K over rounds, showing how ANSD-PP adjusts
2. `prefetch_efficiency_heatmap.png`: 2D heatmap (RTT × BW) showing prefetch hit rate
3. `early_exit_savings.png`: Bar chart of compute time saved by early aborts

**New leaderboard columns:**
- "Avg Adaptive K": Mean K selected by ANSD-PP
- "Prefetch Hit Rate": % of prefetched drafts that were used
- "Early Abort Rate": % of rounds aborted early
- "Robust Speedup": Speedup in worst p95 scenario (tests robustness)

---

## Summary

This proposal presents a comprehensive benchmark for cloud-edge speculative decoding in Ianvs, with a novel **Adaptive Network-Aware Speculative Decoding with Preemptive Pipelining (ANSD-PP)** algorithm. ANSD-PP addresses the core challenges of cloud-edge LLM inference through:

1. **Dynamic K adaptation** to avoid bandwidth saturation and compute waste
2. **Preemptive pipelined drafting** to hide network latency
3. **Early exit signaling** to prevent edge compute waste on doomed speculations

The benchmark provides rigorous evaluation across 7 RTT levels, 4 bandwidth tiers, 3 task complexities, with p95 metrics and break-even analysis. Expected outcomes:

- **1.5-2× speedup** in high-latency scenarios (RTT>80ms) where vanilla speculative fails
- **Robust performance** degrading gracefully in poor conditions rather than becoming slower
- **Comprehensive Ianvs integration** with reproducible configs, detailed logging, and community artifacts

This work will establish Ianvs as the reference platform for cloud-edge LLM benchmarking and provide actionable guidance for practitioners deploying speculative decoding in real-world edge environments.


## **Transparency & Commitment**
In the spirit of open-source transparency, I want to state that I utilized Large Language Models (LLMs) as a brainstorming partner to refine the structure and technical phrasing of this proposal. However, the core architectural concepts—specifically the ANSD-PP mechanism, the pipelining strategy, and the Ianvs integration design represent my own technical understanding and intent.

I have verified every component of this proposal against the KubeEdge/Ianvs documentation and relevant research papers. I fully understand the implementation challenges involved (particularly asyncio concurrency and network simulation) and am confident in my ability to execute this plan independently during the mentorship. I look forward to the opportunity to turn this proposal into a production-ready benchmark for the community.
