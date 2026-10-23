# Design notes

Four questions, answered against the code in `src/pytrainer/` and the
measurements in `results/sweep/`.

---

## 1. The importance criterion, and why it approximates the loss change

### What the code computes

For every eligible weight $w_i$ (weights only; biases are never pruned),
`GlobalTaylorPruner.observe()` runs once per minibatch $b$, after
`backward()` and before `optimizer.step()`, so $w$ and $g$ come from the same
point:

$$
s_i^{(b)} = \bigl|\, w_i \, g_i^{(b)} \bigr| \qquad g_i^{(b)} = \frac{\partial L_b}{\partial w_i}
$$

and keeps a bias-corrected exponential moving average over minibatches
(decay $\beta = 0.9$):

$$
m_i \leftarrow \beta\, m_i + (1-\beta)\, s_i^{(b)}, \qquad
\hat s_i = \frac{m_i}{1 - \beta^{t}}
$$

Pruning ranks all alive weights of all layers together by $\hat s_i$ and
removes the lowest-scoring ones, down to an exact global count. The other
criteria in the code are `taylor_sq` ($s_i^{(b)} = (w_i g_i^{(b)})^2$) and the
`magnitude` baseline ($|w_i|$).

### Derivation

Removing connection $i$ means changing that weight from $w_i$ to $0$, i.e.
a perturbation $\delta = -w_i e_i$. Expand the loss around the current weights:

$$
\Delta L_i = L(w - w_i e_i) - L(w)
= -\,g_i w_i \;+\; \tfrac12 H_{ii} w_i^2 \;+\; O(w_i^3)
$$

**First-order term.** $-g_i w_i$ is the linear prediction of the loss change.
The quantity that matters for ranking is its magnitude: a connection whose
removal moves the loss a lot either way is important. This gives the first-order
Taylor criterion $|w_i g_i|$ (Molchanov et al., 2017/2019).

**Why average $|w g|$ per minibatch rather than take $|w \bar g|$.** During
training, and especially near a minimum, the full-batch gradient $\bar g_i$
tends to zero, so $|w_i \bar g_i|$ says almost nothing. The minibatch gradients
$g_i^{(b)}$ do not vanish; their spread is exactly what the curvature term
measures. For cross-entropy, the diagonal Hessian is approximated by the
empirical Fisher:

$$
H_{ii} \approx F_{ii} = \mathbb{E}_b\!\left[\bigl(g_i^{(b)}\bigr)^2\right]
\;\;\Longrightarrow\;\;
\tfrac12 H_{ii} w_i^2 \approx \tfrac12\, \mathbb{E}_b\!\left[(w_i g_i^{(b)})^2\right]
$$

So at a minimum the expected loss change from removing $w_i$ is (up to the
factor ½) $\mathbb{E}_b[(w_i g_i^{(b)})^2]$, which is the `taylor_sq` criterion:
Optimal Brain Damage with the Fisher in place of the Hessian. The default
`taylor_abs` uses $\mathbb{E}_b|w_i g_i^{(b)}|$, a monotone-ish, outlier-robust
proxy for the same quantity: by Jensen it is at most
$\sqrt{\mathbb{E}_b[(w_i g_i^{(b)})^2]}$. Taking $|\cdot|$ or $(\cdot)^2$
**per minibatch, before averaging**, is what lets the score capture gradient
variance instead of cancelling it. The code does this
(`np.abs(wg)` / `wg * wg` before the EMA), and a unit test checks that gradients
$+1$ and $-1$ give score 1, not 0.

So one quantity covers both regimes. Early in training it tracks the
first-order term; near convergence its square approximates the second-order
(Fisher) term. That is the loss change from deleting the connection.

### Why an EMA, why bias correction

- One minibatch is a noisy estimate of $\mathbb{E}_b$; the EMA averages over
  about $1/(1-\beta) = 10$ recent minibatches. That is recent enough to track the
  weights as they move, and smooth enough to rank on.
- Without the $1/(1-\beta^t)$ correction, the first scores would be shrunk
  toward 0 by the zero initialization. Ranking across layers is not affected
  (every layer has the same $t$), but the scores would no longer be on the
  scale of $|w g|$.
- Pruned weights score 0: their gradient is 0 (see §2), and `observe()` also
  masks them explicitly.

### Assumptions, and what the code does about them

| approximation | where it breaks | mitigation in this code |
|---|---|---|
| Second-order expansion ($w_i$ "small") | removing a large weight | gradual cubic schedule: each step removes few weights, then retrains |
| Diagonal: each removal judged alone | removing many correlated weights at once ignores $H_{ij}$ cross-terms | same: pruning happens every epoch over 70 epochs, so cross-effects get re-measured after each step |
| Empirical Fisher ≈ Hessian | far from a minimum | 10 dense warm-up epochs before the first prune |
| Scores from past weights | weights keep moving | EMA horizon ≈ 10 minibatches; fine-tuning after pruning |

### What the evidence says (`results/sweep/`)

The derivation predicts something magnitude cannot see. A weight on a path that
no longer reaches the output gets $g_i \equiv 0$, so it scores 0 and goes first,
which is correct because removing it costs nothing. Magnitude keeps it if it is
large.

The sweep shows exactly that. At 95% sparsity, global magnitude keeps fc1's
large weights while fc2 is cut to its 2% floor, so the survivors no longer form
input-to-output paths. The effective networks shrink to 2-1-1-2, and in 2 of 30
seeds to 2-0-0-2: a constant output and 50% accuracy. Taylor keeps connected
paths, typically 2-8-4-2, and reaches 96.6% versus magnitude's 81.0% over 30
seeds. At 90% the gap is +0.26 pp (CI +0.18 to +0.35), and it held on 30
held-out seeds.

---

## 2. What "the gradient of a masked weight" is, and why

### What the engine computes

`Linear` computes $Y = X\,(W \odot M) + b$, with the mask $M$ a `Tensor` that
has `requires_grad=False` and is not returned by `parameters()`. The product
$W \odot M$ is an ordinary `ops.mul` node, so the backward pass is just the
product rule. Its backward is
`(a, lambda g: _unbroadcast(g * b.data, a.shape))`, which gives:

$$
\frac{\partial L}{\partial W} = M \odot \frac{\partial L}{\partial W_{\text{eff}}},
\qquad W_{\text{eff}} = W \odot M,
\qquad \frac{\partial L}{\partial M}\ \text{not computed}
$$

**The gradient of a masked weight is exactly 0.** Alive weights get the ordinary
gradient, unchanged: with $M_{ij} = 1$ the product is bit-identical to the
dense case, which a unit test checks. There is no special case anywhere;
the engine never knows a mask exists.

### Why this is the right choice

1. **It is the true derivative of the function being trained.** The loss
   depends on $W_{ij}$ only through $W_{ij} M_{ij}$. When $M_{ij}=0$, the loss is
   constant in $W_{ij}$, so $\partial L / \partial W_{ij} = 0$ is not an
   approximation or a convention. `test_layers.py` sets every pruned raw weight
   to $10^6$ and checks that the output is bit-identical. That test still calls
   the removed `set_weight_mask` helper and needs updating along with the
   other failing tests.
2. **It matches the pruning semantics.** The pruner is monotonic: pruned
   connections never regrow. A zero gradient means nothing pushes a pruned
   weight to change, and it keeps the saliency of pruned weights at 0, so they
   can never be re-ranked as important.
3. **It keeps the mask out of optimization structurally.** $M$ is a constant
   leaf: no gradient is ever stored on it, and Adam is never given it. A test
   checks that registering a non-grad tensor as a parameter fails.
4. **It is exact for the saliency criterion.** For alive weights,
   $\partial L/\partial W = \partial L/\partial W_{\text{eff}}$, so the Taylor
   score uses the real sensitivity of the network as pruned so far, not of the
   dense network it started as.

### What a zero gradient does *not* guarantee, and how the code covers it

A zero gradient is not enough with Adam. If a weight built up momentum
$m_{t-1} \ne 0$ before it was pruned, the update
$\eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)$ keeps moving it even when
$g_t = 0$. A test demonstrates this with plain Adam. So:

- `MaskedAdam.set_mask` is called by the pruner after every prune. It zeroes
  the pruned raw weights, their gradients, **and their $m$ and $v$**.
- After that the update is exactly $0 / (0 + \epsilon) = 0$ on every step, in
  floating point, not just approximately.
- `MaskedAdam.step` also zeroes gradients at pruned entries before updating.
  This is defense in depth for gradients set by something other than the engine.
- `pruner.check_integrity(optimizer)` runs every epoch. It asserts that pruned
  weights, gradients, $m$ and $v$ are exactly 0 and that the optimizer's mask
  equals the layer mask.

### The alternative that was rejected

The **straight-through estimator** would pass $\partial L / \partial W_{\text{eff}}$
to pruned weights as if the mask were not there. That is the right choice for
*dynamic* sparse training (RigL-style regrowth), where the gradient at a dead
connection signals whether to revive it. It is the wrong choice here:

- It would make pruned weights drift, so they would no longer be exactly 0.
- It would give pruned weights nonzero saliency, which contradicts monotonic
  pruning.
- It would optimize a function different from the one the network computes.

The quantity it would use, $\partial L / \partial W_{\text{eff}}$, does exist in
the engine as the gradient of the intermediate `effective_weight` node. Adding
regrowth would mean reading it from there, not changing the gradient rule.

---

## 3. Where the engine bottlenecks, and how to optimize it

### Measurements

Training step, batch 64, 2-32-16-2 MLP, `MaskedAdam` and `observe()`, single
thread. The figures are the mean over 440 steps:

| phase | µs / step | share |
|---|---|---|
| forward (incl. loss) | 243 | 44% |
| backward | 175 | 32% |
| optimizer step | 97 | 18% |
| saliency `observe()` | 32 | 6% |
| **total** | **559** | |

The same forward and backward math written as plain NumPy, with no graph,
takes **91 µs**. The engine's forward and backward together (418 µs) are
**about 4.6× the arithmetic**. Each step builds 12 graph nodes (each with a
closure and a `Tensor`) and allocates 18 zero-filled gradient buffers.

**The bottleneck is per-op Python overhead, not FLOPs.** The matrices are
64×32 at most. Every op pays roughly the same fixed cost (Python call, closure,
`Tensor.__init__`'s `np.asarray`, a fresh output array) whether it does 100
or 100,000 FLOPs. Specifically:

1. **Too many nodes per layer.** `Linear` is three nodes: `mul` (W⊙M),
   `matmul`, `add`. Each `add`/`mul` backward calls `_unbroadcast`, which loops
   over dimensions even when no broadcasting happened.
2. **Gradient buffers are zero-filled, then added to.** `_ensure_grad` allocates
   `zeros_like` and then does `+=`. The first contribution could just be
   assigned.
3. **Every intermediate's gradient is kept until the graph dies.** All 12 nodes
   keep `.grad` arrays alive. At this size that is harmless; for large
   activations it is the memory bottleneck.
4. **The graph is rebuilt and re-sorted every step.** The DFS topological sort
   in `backward()` runs each step even though the MLP's graph is identical
   every time.
5. **No inference mode.** `evaluate()` builds a full graph with closures for
   forward passes that never call `backward()`.
6. **float64 throughout.** That is twice the memory traffic of float32 for no
   accuracy benefit here.
7. **The optimizer allocates every step.** `np.where` creates new gradient
   arrays for masked parameters, and the moment updates create temporaries;
   that is 97 µs for only 658 parameters.
8. **Validation in hot paths.** The loss checks target dtype and range, and
   `Adam.step` checks shape and finiteness on every call.

### How to optimize, in order of payoff per line of code

| change | removes | expected effect |
|---|---|---|
| Fused `linear(x, W, M, b)` op with a hand-written backward ($\partial W = M \odot x^\top g$, $\partial b = \sum g$, $\partial x = g (W\odot M)^\top$), optionally fusing ReLU | 2 of 3 nodes per layer, all `_unbroadcast` calls | biggest single win; the node count drops from 12 to 4 |
| Fold the mask into W once per prune (W is already exactly 0 at pruned entries), dropping the per-step `mul` | one elementwise op + node per layer | small, free |
| Assign the first gradient instead of `zeros` + `+=`; free non-leaf `.grad` right after its backward runs | 18 allocations/step; peak memory | allocator pressure, memory ∝ depth |
| `no_grad()` context: ops skip closure and node creation when grad is off | graph cost in evaluation | evaluation becomes plain NumPy |
| Cache the topological order when the graph's structure repeats | DFS per step | small at 12 nodes, linear in graph size |
| float32 | half the bandwidth | ≈ 2× on memory-bound ops at scale |
| Preallocated buffers with `out=` in Adam and the ops | temporaries per step | optimizer step toward its arithmetic cost |
| Debug-only validation (`__debug__` / a flag) | per-call checks | small, constant |

Together these could bring the step close to the 91 µs floor. Past that point,
the remaining overhead is NumPy's per-call cost, and the fixes are structural:

- **Batch the experiment, not the op.** The sweep trains 270 small, independent
  models in separate processes. Stacking the seeds into a batch dimension
  (`(S, in, out)` weights, `np.matmul` broadcasting over S) turns 30 tiny
  matmuls into one larger one, which is the vmap idea. Process parallelism
  scaled poorly here: 8 runs on 8 workers took 5.8 s, against 1.5 s for one
  run alone.
- **Compile the graph.** The static MLP graph could be traced once and run as a
  compiled kernel (Numba, or a JAX/XLA-style trace), removing Python from the
  inner loop entirely.

**Sparsity is not an engine speedup.** Training runs dense-times-mask, so
pruning makes training no cheaper. Section 4 and `results/sweep/README.md` show
that even an inference-only sparse kernel loses to dense BLAS at this size.

---

## 4. Serving a self-pruned model in a multi-tenant inference service

The measurements that drive this design (`results/sweep/cost.csv`):

- **Unstructured sparsity does not pay at small sizes.** A CSC sparse forward
  pass with 10× fewer multiply-adds was 3–5× slower than dense BLAS.
- **Compaction pays.** Removing disconnected units gives a smaller dense network
  that is 1.8× faster at batch 256 and 2.5× at batch 100k (90% sparsity).
- **Batch size dominates.** One request costs about 8 µs regardless of
  sparsity, because that is call overhead. Batch 256 costs 0.1–0.2 µs per
  sample.
- **Pruning can silently break a model.** Global magnitude at 95% produced
  constant-output networks in 2 of 30 seeds.

So the service treats pruning as a **build-time compression step that must
prove itself**, and puts its runtime effort into batching and isolation.

### 1) Build: turn a training checkpoint into a serving artifact

1. **Fold and compact.** Bake $W \odot M$ into the weights; no mask exists at
   runtime. Run `compact()`: delete units with no outgoing connections, and fold
   units with no incoming connections into the next layer's bias. The output is
   a smaller dense model, verified to give identical logits on a fixed
   calibration set. The sweep's tolerance is 1e-9.
2. **Choose the kernel per layer by measurement, not by FLOPs.** For each
   compacted layer, benchmark dense against sparse (CSR or a vendor sparse
   kernel) on the target hardware at the expected batch sizes, and keep the
   faster one. At this model size that is always dense. Unstructured sparsity
   starts to win only for large layers at very high sparsity. On GPUs the
   practical form is **structured** sparsity: whole-unit pruning, which is what
   compaction produces, or 2:4 semi-structured sparsity with hardware support.
3. **Quantize** (int8 with per-channel scales) after compaction. Pruning and
   quantization compound, and the accuracy gate below checks both together.
4. **Promotion gate. This step fails closed.** An artifact is promoted only if:
   - its accuracy on a large held-out evaluation set is within a fixed tolerance
     of the dense model;
   - its graph is not degenerate: every output still has a path from the input;
   - its measured latency and throughput beat the dense artifact at the
     production batch sizes.

   Otherwise the tenant keeps the dense model. The 2-0-0-2 collapse in the
   sweep is exactly what this gate exists to catch. "Claims to be cheaper" has
   to be a measured number in the artifact's metadata.
5. **Artifact metadata.** Store content-addressed weights, architecture after
   compaction, sparsity, evaluation results, benchmark results, and the training
   config and seed for reproducibility.

### 2) Runtime: many tenants, many small models

**Model placement and loading.**
- A model registry maps (tenant, model, version) to an artifact in object
  storage.
- Serving nodes keep hot models in memory under an LRU budget sized in bytes.
  A compacted model like this one is a few KB, so a node holds thousands.
- Cold models are loaded on demand.
- Consistent hashing on (tenant, model) gives each model a small set of home
  replicas, so caches stay warm without every node loading every model.

**Batching is the main throughput lever.**
- *Per-model dynamic batching:* hold requests for up to a few milliseconds or
  until the batch is full, whichever comes first. The window is a per-tenant
  latency/throughput setting.
- *Across tenants:* each tenant's compacted model has a different shape, so
  requests for different models cannot share one matmul directly. Bucket the
  shapes (round each hidden width up to, say, a multiple of 8). Zero-pad models
  into their bucket; padding with zero rows and columns is exact for a ReLU MLP.
  Then serve every model in a bucket with one *batched* matmul, `(models, n, in)
  @ (models, in, out)`. Padding waste is bounded by the bucket granularity. This
  turns thousands of tiny per-tenant calls into a few large ones, which is what
  the batch-1 versus batch-256 numbers say matters most.
- Results are scattered back by request ID. The batched computation is
  separate per model mathematically, but routing is the isolation boundary, so
  it gets tests and per-request checksums.

**Isolation and fairness.**
- Per-tenant rate limits and concurrency quotas, enforced at admission.
- Weighted fair queuing into the batcher, so one tenant's burst cannot fill
  every batch.
- Bulkheads: large or latency-critical tenants get dedicated pools; the long
  tail shares bucketed pools.
- Load shedding by priority under overload, returning explicit 429s rather
  than timing out silently.
- Artifacts and logs are access-controlled per tenant; nothing from one tenant's
  inputs or outputs reaches another's response or metrics.

**Scaling and placement.**
- Autoscale on queue delay and p99 latency, not CPU utilization. Batching
  deliberately raises utilization.
- Small compacted MLPs belong on CPU. Accelerators only pay off for large models
  or very large bucketed batches.
- Keep a dense fallback replica per hot model: if a pruned artifact misbehaves
  in production, traffic moves back in one config change.

### 3) Operations: prove the savings and the quality continuously

- **Rollout.** Shadow traffic first: run the pruned artifact next to the
  current one and compare outputs and agreement rates. Then canary by tenant,
  with automatic rollback on accuracy proxies or latency regressions.
  Everything is versioned, and rollback is a registry pointer change.
- **Cost accounting per tenant.** Record measured compute time per request,
  plus multiply-adds as a secondary figure. Savings are reported against the
  dense artifact under the same batching, because FLOPs are not latency, as
  the sparse-kernel result shows.
- **Drift.** Monitor input distribution and output confidence per tenant.
  Pruned models have less spare capacity, so they can degrade earlier under
  drift than dense ones. A drift alarm triggers re-evaluation against the
  promotion gate, and if needed re-pruning from the latest dense checkpoint.
- **Reproducibility.** Every artifact can be rebuilt from (code version,
  config, seed). This repo's experiments are already deterministic per seed,
  and the sweep relies on that for its paired comparison.
