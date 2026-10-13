# Sampled action inference

The cloud simulator launcher supports an inference-only choice between greedy
and sampled action tokens. It does not alter training, the pretrained weights,
LoRA adapters, the action convention, movement bounds or IK checks.

By default, inference keeps `do_sample=False`. Add these options to an existing
`python -m cloud.cloud_session` command to enable sampling:

```bash
--sample-actions --temperature 1.0 --top-p 1.0
```

`--sample-actions` alone uses temperature 1.0 and top-p 1.0: sample from the
conditional distribution over valid action tokens, with top-k disabled and one
beam. All seven components are generated sequentially, conditioning each on the
previous sampled components. Every new prediction draws again; no seed is reset
on each request. Repeated identical inputs can therefore yield different actions.

Temperature below 1 concentrates the distribution on high-probability tokens.
Top-p below 1 discards its low-probability tail. For example:

```bash
--sample-actions --temperature 0.7 --top-p 0.95
```

That is an experimental setting, not a benchmarked recommendation. If a stationary
action dominates the distribution, concentrating or truncating the distribution
may still produce the same stationary action repeatedly. Sampling can also produce
less useful movements; it is not a replacement for learning a reliable policy.
Compare greedy and sampled trials with the same checkpoint and starting scene,
and repeat sampled trials because one trial does not measure the success rate.

The sampling path masks out text, padding and end-of-sequence tokens: only the
256 action token IDs defined by the loaded OpenVLA model are eligible for the
seven output components. OpenVLA's usual action-bin decoder and dataset-specific
unnormalization remain in use. Greedy decoding keeps the previous token-selection
path without that additional mask.

The sampling request travels from the Mac to the resident server with each image.
The response must confirm the requested settings and action-token filtering; a
missing/mismatched confirmation is rejected before robot execution. Once a capable
server has loaded a checkpoint, greedy and sampled trials can use the same resident
model. A previously running server without sampling support is restarted once
when a sampled session is launched. The existing server runtime limit still applies.
No Docker rebuild is needed; the launcher pulls current repository code.

Each trial saves `generation_options.json` beside its run report. Each
`step_*/prediction.json` records the actual generation settings reported by the
server. Sampled mode is available through `cloud.cloud_session` and
`inference.remote_loop`; the standalone `inference.openvla_mac` command remains
greedy. GPU benchmarking and matched teacher-forced diagnostics remain greedy.

Verification uses the installed Transformers 4.40.1 generation API, mocked model
calls, action-token masking and repeated probability draws, local HTTP requests,
resident-server capability checks, and a simulator trial with mock predictions.
Actual recovery or task-success gains have not yet been measured on a cloud GPU.

Generation reference:
[Hugging Face Transformers 4.40.1 generation strategies](https://huggingface.co/docs/transformers/v4.40.1/en/generation_strategies).
