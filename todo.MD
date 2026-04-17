# TODO

Every important claim should trace cleanly to either a citation, a figure, a
table, or a reproducible experiment. Below we antecipate some hard review questions,
so that we can address them in the paper.

## Hard review questions



1. Lack of Competitive "Physics" Baselines - A demanding reviewer might argue that the paper compares the model to itself rather than the current standard in the field.

  - The Argument: "While 77.98% accuracy is 'statistically credible,' it is significantly lower than current SOTA for HIGGS, which often exceeds 80% or 0.88+ ROC AUC using Deep Sets or advanced Transformers. Without a side-by-side comparison of the NPU's efficiency (samples/s/watt) against a GPU running a SOTA model, the systems case is incomplete." 

  - The "Mean" Critique: "This is essentially a high-speed implementation of an outdated model."

2. "Incomplete" On-Chip Deployment - The fact that the embed and the head are on the CPU is a major target for a systems-focused reviewer.

  - The Argument: "The paper claims a 'NPU-native' design, yet the fastest operating point relies on the host CPU for both the ingress (embed) and egress (head) stages. This introduces a 'hidden' latency and power cost associated with the PCIe/memory bus that isn't fully accounted for in the 'Kernel Throughput' metrics." 

  - The "Mean" Critique: "The NPU is merely a co-processor here; the paper fails to demonstrate a truly standalone spatial dataflow pipeline."

  - Partial response: with the double-buffered host/kernel overlap (§5.2)
    host embed/head/tile/detile work runs fully concealed behind the kernel,
    so the CPU-assisted path is not actually on the critical wall-time path.
    Moving embed/head on-array remains desirable for the "NPU-native"
    narrative but is no longer a wall-throughput fix.

3. Limited Architectural Diversity - A reviewer looking for "ML novelty" might find the residual MLP too simple.

  - The Argument: "The 'conveyor-belt' mapping is specific to a very narrow class of sequential residual MLPs. The paper does not demonstrate how this approach scales to more complex topologies used in modern jet tagging, such as Graph Neural Networks (GNNs) or Attention mechanisms, which are the primary interest of the ML4Jets community." 

  - The "Mean" Critique: "The hardware-to-model mapping is trivial because the model was simplified until it fit the hardware, rather than solving the harder problem of mapping complex models to spatial arrays."

4. Throughput vs. Latency Metrics - In High-Energy Physics, deterministic latency is often more important than aggregate throughput. 

  - The Argument: "The paper reports 'Repeated-sample throughput' over 50M samples. In a real Level-1 trigger environment, we care about the worst-case latency for a single bunch crossing. The paper provides no jitter analysis or $99^{th}$ percentile latency figures, making it impossible to evaluate for real-time trigger suitability."

  - The "Mean" Critique: "High throughput via batching is easy; low-latency, single-sample determinism is what actually matters for triggers."

5. Not using the full NPU

  The Argument: "On the 8 Layer MLP, you leave 3/4 of the compute tiles of the NPU unused. Could you use them to launch 4 parallel instances of the model and get 4x the throughput?" 

  - The "Mean" Critique: "you left performance on the table, and it was low hanging fruit"

6 - No results for 16 layer MLP

  The Argument: "You only show results for the 8 and 32 layer MLPs. Why not show results for the 16 layer MLP?"     

  - The "Mean" Critique: "you didn't fully explore the design space"

7 - Gap with kernel throughput  [ADDRESSED §5.2]

  The Argument: "The paper claims a 'NPU-native' design, yet there seems to be a gap between the kernel throughput and the wall time throughput. How can you justify this gap? And how could you close it? If you could achieve 30M samples/s on the kernel, why not 30M samples/s on the wall time?"

  - The "Mean" Critique: "you have bottlenecked the NPU with a suboptimal software implementation"

  - Status: partially resolved by the stacked-host + double-buffered pipeline
    (see `resmlp/streaming_infer.py` and `resmlp/streaming_op.py`). With the
    corrected isolated-kernel calibration, the two `H=32` retained
    configurations land at about 97--99\% of kernel throughput, but the wider
    `H=64, L=32` accuracy model still sits around 73\%. Table 4 and §5.2 now
    report sync wall, async wall, isolated kernel, and the async/kernel ratio.

8 - Comparison with the FPGA recent paper

  - The argument: The paper " Development and demonstration of the CMS Phase-2
Level-1 trigger Data Scouting baseline system for HL-LHC" available at https://inspirehep.net/files/1a1f1ae803ff794879636eb00f8d7e7b seems to describe the state of the art in FPGA based L1 trigger.  How do you compare?

  - The "Mean" Critique: "you are not comparing with the state of the art"
