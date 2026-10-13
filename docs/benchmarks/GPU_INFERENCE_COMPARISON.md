# GPU inference comparison

One request = one RGB image + instruction → one seven-value action. Inference batch size is 1.

Timings are seconds. The first 5 recorded requests per trial are excluded as warmup. Short trials without remaining samples show missing timings. These are observed simulator runs, not identical-input hardware benchmarks; model/checkpoint, instruction, network, Mac, storage and software differences can affect results. Rejected/held actions remain prediction requests. Combined benchmarks start with 5 warmup requests and 20 measured inference requests, both on the cloud and from the Mac when connected. Cloud-loopback RTT is never reported as Mac latency. When Mac measurements exist, the server breakdown uses those same responses to keep RTT and server timing paired; otherwise it uses the cloud samples. Both raw sets are retained.


## Average by GPU type

Means are weighted by measured request counts, not an average of rounded table values. There is one row per reported GPU type: all A40 runs are combined, for example. These overall averages mix VRAM/ECC configurations, model checkpoints, software versions, inputs, and simulator/benchmark runs; use the detailed tables below to inspect those differences. They describe our observed runs rather than an isolated hardware ranking. Local timing is Mac-dependent. Each metric includes only runs that recorded it; cloud-only runs contribute server timing but not Mac RTT or simulator time. The per-run table below preserves each observation.

| GPU | Runs | Measured requests | Mean cycle s | Mean request RTT s | Mean prediction s | USD/hour | 100 requests USD | 100 simulator cycles USD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVIDIA A40 | 65 | 2254 | 1.516 | 1.015 | 0.325 | 0.59 | 0.0166 | 0.0248 |
| NVIDIA RTX A5000 | 1 | 20 | — | 1.347 | 0.337 | 0.27 | 0.0101 | — |

## Full cycle and estimated cost for 100 steps

| GPU | Run | Measured/all requests | Full cycle s | Request RTT s | Request p95 s | Model prediction s | USD/hour | 100 cycles s | 100 active cycles USD | 100 cycles + preparation/loading USD | Type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVIDIA A40 | simulator_trial | 0/5 | — | — | — | — | 0.59 | — | — | — | Simulator |
| NVIDIA A40 | remote_loop_20261010_060952_21db409c | 95/100 | 1.464 | 0.960 | 1.196 | 0.319 | 0.59 | 146.4 | 0.0240 | 0.0345 | Simulator |
| NVIDIA A40 | remote_loop_20261010_072519_5186f692 | 95/100 | 1.416 | 0.911 | 0.981 | 0.299 | 0.59 | 141.6 | 0.0232 | 0.0301 | Simulator |
| NVIDIA A40 | remote_loop_20261010_133609_315d8199 | 82/87 | 1.630 | 1.123 | 1.341 | 0.331 | 0.59 | 163.0 | 0.0267 | 0.0276 | Simulator |
| NVIDIA A40 | remote_loop_20261011_003638_4fe3675c | 95/100 | 1.561 | 1.053 | 1.323 | 0.321 | 0.59 | 156.1 | 0.0256 | 0.0265 | Simulator |
| NVIDIA A40 | remote_loop_20261011_021037_83d8c4a3 | 40/45 | 1.450 | 0.944 | 1.034 | 0.308 | 0.59 | 145.0 | 0.0238 | 0.0248 | Simulator |
| NVIDIA A40 | remote_loop_20261011_023950_e87092ab | 57/62 | 1.865 | 1.357 | 2.508 | 0.346 | 0.59 | 186.5 | 0.0306 | 0.0315 | Simulator |
| NVIDIA A40 | remote_loop_20261011_130030_90592976 | 95/100 | 1.411 | 0.905 | 1.030 | 0.295 | 0.59 | 141.1 | 0.0231 | 0.0240 | Simulator |
| NVIDIA A40 | remote_loop_20261011_130522_3943abad | 95/100 | 1.421 | 0.920 | 1.045 | 0.292 | 0.59 | 142.1 | 0.0233 | 0.0243 | Simulator |
| NVIDIA A40 | remote_loop_20261012_114419_a61c5550 | 108/113 | 1.566 | 1.060 | 1.214 | 0.343 | 0.59 | 156.6 | 0.0257 | 0.0268 | Simulator |
| NVIDIA A40 | remote_loop_20261012_115022_efd46fc5 | 36/41 | 1.524 | 1.016 | 1.149 | 0.343 | 0.59 | 152.4 | 0.0250 | 0.0261 | Simulator |
| NVIDIA A40 | remote_loop_20261012_115331_fcdda8c6 | 19/24 | 1.515 | 1.010 | 1.227 | 0.347 | 0.59 | 151.5 | 0.0248 | 0.0260 | Simulator |
| NVIDIA A40 | remote_loop_20261012_122158_a28ecf48 | 9/14 | 1.498 | 0.995 | 1.106 | 0.336 | 0.59 | 149.8 | 0.0246 | 0.0255 | Simulator |
| NVIDIA A40 | remote_loop_20261012_122230_1398de3f | 145/150 | 1.515 | 1.011 | 1.342 | 0.323 | 0.59 | 151.5 | 0.0248 | 0.0258 | Simulator |
| NVIDIA A40 | remote_loop_20261012_122634_3ccc119e | 20/25 | 1.456 | 0.965 | 1.138 | 0.324 | 0.59 | 145.6 | 0.0239 | 0.0248 | Simulator |
| NVIDIA A40 | remote_loop_20261012_122746_afdcf70c | 15/20 | 1.451 | 0.951 | 1.035 | 0.325 | 0.59 | 145.1 | 0.0238 | 0.0247 | Simulator |
| NVIDIA A40 | remote_loop_20261012_122831_de4972a4 | 30/35 | 1.554 | 1.047 | 1.339 | 0.320 | 0.59 | 155.4 | 0.0255 | 0.0264 | Simulator |
| NVIDIA A40 | remote_loop_20261013_004429_0ac8a961 | 5/10 | 1.776 | 1.266 | 1.467 | 0.344 | 0.59 | 177.6 | 0.0291 | 0.0303 | Simulator |
| NVIDIA A40 | remote_loop_20261013_004457_d1ae30e2 | 3/8 | 1.659 | 1.155 | 1.193 | 0.323 | 0.59 | 165.9 | 0.0272 | 0.0284 | Simulator |
| NVIDIA A40 | remote_loop_20261013_004516_7b5da887 | 1/6 | 1.678 | 1.175 | 1.175 | 0.297 | 0.59 | 167.8 | 0.0275 | 0.0287 | Simulator |
| NVIDIA A40 | remote_loop_20261013_020816_48c2152f | 20/25 | 1.394 | 0.952 | 1.092 | 0.339 | 0.59 | 139.4 | 0.0229 | 0.0240 | Simulator |
| NVIDIA A40 | remote_loop_20261013_020901_f64b721f | 29/34 | 1.491 | 0.986 | 1.100 | 0.333 | 0.59 | 149.1 | 0.0244 | 0.0255 | Simulator |
| NVIDIA A40 | remote_loop_20261013_021001_2a2f89e2 | 55/60 | 1.512 | 1.021 | 1.264 | 0.325 | 0.59 | 151.2 | 0.0248 | 0.0259 | Simulator |
| NVIDIA A40 | remote_loop_20261013_021421_598d0987 | 95/100 | 1.590 | 1.085 | 1.432 | 0.343 | 0.59 | 159.0 | 0.0261 | 0.0276 | Simulator |
| NVIDIA A40 | remote_loop_20261013_021707_d8a114c5 | 33/38 | 1.542 | 1.076 | 1.342 | 0.336 | 0.59 | 154.2 | 0.0253 | 0.0268 | Simulator |
| NVIDIA A40 | remote_loop_20261013_022040_8ed20480 | 95/100 | 1.525 | 1.020 | 1.311 | 0.336 | 0.59 | 152.5 | 0.0250 | 0.0264 | Simulator |
| NVIDIA A40 | remote_loop_20261013_022319_a6d8b143 | 21/26 | 1.489 | 0.986 | 1.113 | 0.344 | 0.59 | 148.9 | 0.0244 | 0.0258 | Simulator |
| NVIDIA A40 | remote_loop_20261013_022403_5dc12f75 | 86/91 | 1.497 | 0.993 | 1.183 | 0.341 | 0.59 | 149.7 | 0.0245 | 0.0259 | Simulator |
| NVIDIA A40 | remote_loop_20261013_022854_75e43126 | 41/46 | 1.587 | 1.082 | 1.404 | 0.337 | 0.59 | 158.7 | 0.0260 | 0.0272 | Simulator |
| NVIDIA A40 | remote_loop_20261013_023013_9609f73d | 31/36 | 1.600 | 1.111 | 1.320 | 0.358 | 0.59 | 160.0 | 0.0262 | 0.0274 | Simulator |
| NVIDIA A40 | remote_loop_20261013_023115_96815de3 | 95/100 | 1.567 | 1.062 | 1.270 | 0.336 | 0.59 | 156.7 | 0.0257 | 0.0269 | Simulator |
| NVIDIA A40 | remote_loop_20261013_024046_7faaeb28 | 70/75 | 1.555 | 1.051 | 1.377 | 0.327 | 0.59 | 155.5 | 0.0255 | 0.0264 | Simulator |
| NVIDIA A40 | remote_loop_20261013_024249_f3af7673 | 86/91 | 1.569 | 1.066 | 1.423 | 0.332 | 0.59 | 156.9 | 0.0257 | 0.0266 | Simulator |
| NVIDIA A40 | remote_loop_20261013_024522_2e573d8d | 8/13 | 1.546 | 1.048 | 1.176 | 0.345 | 0.59 | 154.6 | 0.0253 | 0.0263 | Simulator |
| NVIDIA A40 | remote_loop_20261013_040917_acebfb07 | 16/21 | 1.391 | 0.887 | 1.004 | 0.306 | 0.59 | 139.1 | 0.0228 | 0.0237 | Simulator |
| NVIDIA A40 | remote_loop_20261013_040954_239e0172 | 16/21 | 1.409 | 0.903 | 1.031 | 0.312 | 0.59 | 140.9 | 0.0231 | 0.0240 | Simulator |
| NVIDIA A40 | remote_loop_20261013_041030_f12b9235 | 17/22 | 1.422 | 0.919 | 1.077 | 0.299 | 0.59 | 142.2 | 0.0233 | 0.0243 | Simulator |
| NVIDIA A40 | remote_loop_20261013_041321_8805e94b | 17/22 | 1.493 | 0.991 | 1.121 | 0.333 | 0.59 | 149.3 | 0.0245 | 0.0254 | Simulator |
| NVIDIA A40 | remote_loop_20261013_041412_8a739ca4 | 18/23 | 1.503 | 0.998 | 1.212 | 0.327 | 0.59 | 150.3 | 0.0246 | 0.0256 | Simulator |
| NVIDIA A40 | remote_loop_20261013_041455_1cbdf9b8 | 17/22 | 1.440 | 0.938 | 1.022 | 0.301 | 0.59 | 144.0 | 0.0236 | 0.0246 | Simulator |
| NVIDIA A40 | remote_loop_20261013_041734_b4164a41 | 17/22 | 1.511 | 1.007 | 1.114 | 0.345 | 0.59 | 151.1 | 0.0248 | 0.0257 | Simulator |
| NVIDIA A40 | remote_loop_20261013_041813_f370b0c6 | 17/22 | 1.493 | 0.989 | 1.208 | 0.313 | 0.59 | 149.3 | 0.0245 | 0.0254 | Simulator |
| NVIDIA A40 | remote_loop_20261013_041850_e86626e6 | 17/22 | 1.485 | 0.980 | 1.103 | 0.325 | 0.59 | 148.5 | 0.0243 | 0.0252 | Simulator |
| NVIDIA A40 | remote_loop_20261013_042244_e5d85e46 | 17/22 | 1.454 | 0.950 | 1.060 | 0.321 | 0.59 | 145.4 | 0.0238 | 0.0248 | Simulator |
| NVIDIA A40 | remote_loop_20261013_042322_289a8f5f | 16/21 | 1.435 | 0.933 | 1.008 | 0.309 | 0.59 | 143.5 | 0.0235 | 0.0244 | Simulator |
| NVIDIA A40 | remote_loop_20261013_042402_f71b51e0 | 13/18 | 1.422 | 0.917 | 1.037 | 0.296 | 0.59 | 142.2 | 0.0233 | 0.0242 | Simulator |
| NVIDIA A40 | remote_loop_20261013_042546_0a547490 | 11/16 | 1.457 | 0.952 | 1.224 | 0.302 | 0.59 | 145.7 | 0.0239 | 0.0248 | Simulator |
| NVIDIA A40 | remote_loop_20261013_042618_7f5c0bb6 | 13/18 | 1.426 | 0.926 | 0.987 | 0.326 | 0.59 | 142.6 | 0.0234 | 0.0243 | Simulator |
| NVIDIA A40 | remote_loop_20261013_042935_ac686ee3 | 13/18 | 1.463 | 0.965 | 1.063 | 0.320 | 0.59 | 146.3 | 0.0240 | 0.0250 | Simulator |
| NVIDIA A40 | remote_loop_20261013_043008_c17a4b95 | 14/19 | 1.485 | 0.988 | 1.194 | 0.329 | 0.59 | 148.5 | 0.0243 | 0.0254 | Simulator |
| NVIDIA A40 | remote_loop_20261013_043040_fa0cd102 | 12/17 | 1.572 | 1.072 | 1.652 | 0.308 | 0.59 | 157.2 | 0.0258 | 0.0268 | Simulator |
| NVIDIA A40 | remote_loop_20261013_043307_12ea3445 | 14/19 | 1.430 | 0.928 | 1.012 | 0.313 | 0.59 | 143.0 | 0.0234 | 0.0244 | Simulator |
| NVIDIA A40 | remote_loop_20261013_043343_b2368bf0 | 12/17 | 1.476 | 0.973 | 1.185 | 0.307 | 0.59 | 147.6 | 0.0242 | 0.0252 | Simulator |
| NVIDIA A40 | remote_loop_20261013_043412_4c8ece6c | 12/17 | 1.433 | 0.933 | 1.016 | 0.319 | 0.59 | 143.3 | 0.0235 | 0.0245 | Simulator |
| NVIDIA A40 | remote_loop_20261013_043611_4395c010 | 14/19 | 1.476 | 0.975 | 1.115 | 0.332 | 0.59 | 147.6 | 0.0242 | 0.0252 | Simulator |
| NVIDIA A40 | remote_loop_20261013_043651_1d69c320 | 0/4 | — | — | — | — | 0.59 | — | — | — | Simulator |
| NVIDIA A40 | remote_loop_20261013_043712_3ca44d08 | 16/21 | 1.428 | 0.923 | 0.995 | 0.308 | 0.59 | 142.8 | 0.0234 | 0.0244 | Simulator |
| NVIDIA A40 | remote_loop_20261013_043816_dfd3e0a0 | 10/15 | 1.476 | 0.975 | 1.046 | 0.348 | 0.59 | 147.6 | 0.0242 | 0.0252 | Simulator |
| NVIDIA A40 | remote_loop_20261013_044101_933eca37 | 14/19 | 1.396 | 0.892 | 1.039 | 0.299 | 0.59 | 139.6 | 0.0229 | 0.0238 | Simulator |
| NVIDIA A40 | remote_loop_20261013_044141_7fb563b3 | 12/17 | 1.455 | 0.954 | 1.194 | 0.293 | 0.59 | 145.5 | 0.0239 | 0.0248 | Simulator |
| NVIDIA A40 | remote_loop_20261013_044211_e719a9ca | 12/17 | 1.406 | 0.912 | 0.962 | 0.293 | 0.59 | 140.6 | 0.0231 | 0.0240 | Simulator |
| NVIDIA A40 | remote_loop_20261013_044250_c01c5d9f | 16/21 | 1.441 | 0.936 | 1.056 | 0.306 | 0.59 | 144.1 | 0.0236 | 0.0245 | Simulator |
| NVIDIA A40 | remote_loop_20261013_044525_7990fd2e | 14/19 | 1.476 | 0.973 | 1.049 | 0.312 | 0.59 | 147.6 | 0.0242 | 0.0252 | Simulator |
| NVIDIA A40 | remote_loop_20261013_044641_06df011b | 17/22 | 1.449 | 0.944 | 1.029 | 0.322 | 0.59 | 144.9 | 0.0238 | 0.0248 | Simulator |
| NVIDIA A40 | 20261012_040655_255759 | 20/25 | — | 1.261 | 1.471 | 0.325 | 0.59 | 126.1 | 0.0207 | — | Benchmark (Mac) |
| NVIDIA RTX A5000 | 20261012_043822_113279 | 20/25 | — | 1.347 | 1.555 | 0.337 | 0.27 | 134.7 | 0.0101 | — | Benchmark (Mac) |
| NVIDIA RTX A6000 | Not measured | 0 | — | — | — | — | 0.59 | — | — | — | — |

Cost = estimated seconds ÷ 3600 × user-confirmed hourly rate. 100-cycle estimates extrapolate the measured mean, including partial trials; they do not assert task success. Setup-inclusive cost adds recorded model preparation and loading once. It excludes image deployment, adapter merge/export unless covered by recorded preparation, other setup, manual pauses, idle time, shutdown delays and storage charges. Rates are comparison inputs, not historical billing records. Multiple trials can reuse one model loading. For benchmark rows without simulator movements, 100-cycle columns mean 100 requests only: local IK/movement/capture time is unmeasured, not zero. Cloud-only rows cannot estimate Mac RTT costs. Missing measurements stay missing.

## Remote request breakdown — averages in seconds

| Run | Request RTT | Receive body | Decode image | Processor + GPU transfer | Model prediction | Server total | Remaining server overhead | RTT minus server total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simulator_trial | — | — | — | — | — | — | — | — |
| remote_loop_20261010_060952_21db409c | 0.960 | 0.014 | 0.010 | 0.043 | 0.319 | 0.386 | 0.000 | 0.575 |
| remote_loop_20261010_072519_5186f692 | 0.911 | 0.007 | 0.009 | 0.027 | 0.299 | 0.342 | 0.000 | 0.569 |
| remote_loop_20261010_133609_315d8199 | 1.123 | 0.009 | 0.011 | 0.092 | 0.331 | 0.443 | 0.000 | 0.680 |
| remote_loop_20261011_003638_4fe3675c | 1.053 | 0.006 | 0.009 | 0.111 | 0.321 | 0.446 | 0.000 | 0.607 |
| remote_loop_20261011_021037_83d8c4a3 | 0.944 | 0.007 | 0.007 | 0.029 | 0.308 | 0.351 | 0.000 | 0.593 |
| remote_loop_20261011_023950_e87092ab | 1.357 | 0.204 | 0.010 | 0.120 | 0.346 | 0.680 | 0.000 | 0.677 |
| remote_loop_20261011_130030_90592976 | 0.905 | 0.007 | 0.006 | 0.025 | 0.295 | 0.333 | 0.000 | 0.572 |
| remote_loop_20261011_130522_3943abad | 0.920 | 0.006 | 0.008 | 0.030 | 0.292 | 0.336 | 0.000 | 0.584 |
| remote_loop_20261012_114419_a61c5550 | 1.060 | 0.020 | 0.011 | 0.071 | 0.343 | 0.445 | 0.000 | 0.615 |
| remote_loop_20261012_115022_efd46fc5 | 1.016 | 0.016 | 0.012 | 0.063 | 0.343 | 0.434 | 0.000 | 0.583 |
| remote_loop_20261012_115331_fcdda8c6 | 1.010 | 0.011 | 0.011 | 0.076 | 0.347 | 0.444 | 0.000 | 0.566 |
| remote_loop_20261012_122158_a28ecf48 | 0.995 | 0.016 | 0.011 | 0.064 | 0.336 | 0.426 | 0.000 | 0.569 |
| remote_loop_20261012_122230_1398de3f | 1.011 | 0.012 | 0.010 | 0.045 | 0.323 | 0.391 | 0.000 | 0.620 |
| remote_loop_20261012_122634_3ccc119e | 0.965 | 0.010 | 0.010 | 0.036 | 0.324 | 0.380 | 0.000 | 0.585 |
| remote_loop_20261012_122746_afdcf70c | 0.951 | 0.012 | 0.009 | 0.041 | 0.325 | 0.388 | 0.000 | 0.564 |
| remote_loop_20261012_122831_de4972a4 | 1.047 | 0.037 | 0.011 | 0.040 | 0.320 | 0.408 | 0.000 | 0.639 |
| remote_loop_20261013_004429_0ac8a961 | 1.266 | 0.010 | 0.012 | 0.334 | 0.344 | 0.700 | 0.000 | 0.567 |
| remote_loop_20261013_004457_d1ae30e2 | 1.155 | 0.007 | 0.011 | 0.227 | 0.323 | 0.568 | 0.000 | 0.587 |
| remote_loop_20261013_004516_7b5da887 | 1.175 | 0.004 | 0.009 | 0.192 | 0.297 | 0.503 | 0.000 | 0.672 |
| remote_loop_20261013_020816_48c2152f | 0.952 | 0.008 | 0.011 | 0.041 | 0.339 | 0.399 | 0.000 | 0.553 |
| remote_loop_20261013_020901_f64b721f | 0.986 | 0.008 | 0.011 | 0.061 | 0.333 | 0.413 | 0.000 | 0.573 |
| remote_loop_20261013_021001_2a2f89e2 | 1.021 | 0.007 | 0.010 | 0.069 | 0.325 | 0.411 | 0.000 | 0.610 |
| remote_loop_20261013_021421_598d0987 | 1.085 | 0.007 | 0.009 | 0.077 | 0.343 | 0.436 | 0.000 | 0.650 |
| remote_loop_20261013_021707_d8a114c5 | 1.076 | 0.006 | 0.009 | 0.088 | 0.336 | 0.440 | 0.000 | 0.636 |
| remote_loop_20261013_022040_8ed20480 | 1.020 | 0.007 | 0.011 | 0.071 | 0.336 | 0.424 | 0.000 | 0.597 |
| remote_loop_20261013_022319_a6d8b143 | 0.986 | 0.006 | 0.010 | 0.064 | 0.344 | 0.424 | 0.000 | 0.562 |
| remote_loop_20261013_022403_5dc12f75 | 0.993 | 0.006 | 0.009 | 0.083 | 0.341 | 0.439 | 0.000 | 0.554 |
| remote_loop_20261013_022854_75e43126 | 1.082 | 0.007 | 0.010 | 0.093 | 0.337 | 0.447 | 0.000 | 0.634 |
| remote_loop_20261013_023013_9609f73d | 1.111 | 0.007 | 0.009 | 0.135 | 0.358 | 0.509 | 0.000 | 0.602 |
| remote_loop_20261013_023115_96815de3 | 1.062 | 0.007 | 0.009 | 0.064 | 0.336 | 0.416 | 0.000 | 0.647 |
| remote_loop_20261013_024046_7faaeb28 | 1.051 | 0.007 | 0.010 | 0.083 | 0.327 | 0.426 | 0.000 | 0.625 |
| remote_loop_20261013_024249_f3af7673 | 1.066 | 0.006 | 0.009 | 0.094 | 0.332 | 0.442 | 0.000 | 0.624 |
| remote_loop_20261013_024522_2e573d8d | 1.048 | 0.007 | 0.010 | 0.046 | 0.345 | 0.408 | 0.000 | 0.640 |
| remote_loop_20261013_040917_acebfb07 | 0.887 | 0.006 | 0.011 | 0.030 | 0.306 | 0.354 | 0.000 | 0.534 |
| remote_loop_20261013_040954_239e0172 | 0.903 | 0.006 | 0.011 | 0.038 | 0.312 | 0.366 | 0.000 | 0.537 |
| remote_loop_20261013_041030_f12b9235 | 0.919 | 0.005 | 0.010 | 0.032 | 0.299 | 0.346 | 0.000 | 0.573 |
| remote_loop_20261013_041321_8805e94b | 0.991 | 0.006 | 0.009 | 0.033 | 0.333 | 0.382 | 0.000 | 0.609 |
| remote_loop_20261013_041412_8a739ca4 | 0.998 | 0.005 | 0.009 | 0.069 | 0.327 | 0.410 | 0.000 | 0.588 |
| remote_loop_20261013_041455_1cbdf9b8 | 0.938 | 0.006 | 0.009 | 0.031 | 0.301 | 0.347 | 0.000 | 0.590 |
| remote_loop_20261013_041734_b4164a41 | 1.007 | 0.006 | 0.010 | 0.048 | 0.345 | 0.409 | 0.000 | 0.598 |
| remote_loop_20261013_041813_f370b0c6 | 0.989 | 0.006 | 0.009 | 0.039 | 0.313 | 0.367 | 0.000 | 0.622 |
| remote_loop_20261013_041850_e86626e6 | 0.980 | 0.006 | 0.010 | 0.035 | 0.325 | 0.375 | 0.000 | 0.605 |
| remote_loop_20261013_042244_e5d85e46 | 0.950 | 0.006 | 0.010 | 0.040 | 0.321 | 0.377 | 0.000 | 0.573 |
| remote_loop_20261013_042322_289a8f5f | 0.933 | 0.006 | 0.011 | 0.041 | 0.309 | 0.367 | 0.000 | 0.566 |
| remote_loop_20261013_042402_f71b51e0 | 0.917 | 0.006 | 0.011 | 0.035 | 0.296 | 0.348 | 0.000 | 0.569 |
| remote_loop_20261013_042546_0a547490 | 0.952 | 0.006 | 0.011 | 0.032 | 0.302 | 0.350 | 0.000 | 0.602 |
| remote_loop_20261013_042618_7f5c0bb6 | 0.926 | 0.006 | 0.010 | 0.030 | 0.326 | 0.374 | 0.000 | 0.553 |
| remote_loop_20261013_042935_ac686ee3 | 0.965 | 0.007 | 0.011 | 0.048 | 0.320 | 0.386 | 0.000 | 0.579 |
| remote_loop_20261013_043008_c17a4b95 | 0.988 | 0.006 | 0.010 | 0.037 | 0.329 | 0.383 | 0.000 | 0.604 |
| remote_loop_20261013_043040_fa0cd102 | 1.072 | 0.078 | 0.009 | 0.031 | 0.308 | 0.426 | 0.000 | 0.646 |
| remote_loop_20261013_043307_12ea3445 | 0.928 | 0.007 | 0.010 | 0.030 | 0.313 | 0.360 | 0.000 | 0.568 |
| remote_loop_20261013_043343_b2368bf0 | 0.973 | 0.006 | 0.010 | 0.089 | 0.307 | 0.412 | 0.000 | 0.561 |
| remote_loop_20261013_043412_4c8ece6c | 0.933 | 0.007 | 0.010 | 0.039 | 0.319 | 0.374 | 0.000 | 0.559 |
| remote_loop_20261013_043611_4395c010 | 0.975 | 0.007 | 0.010 | 0.034 | 0.332 | 0.383 | 0.000 | 0.592 |
| remote_loop_20261013_043651_1d69c320 | — | — | — | — | — | — | — | — |
| remote_loop_20261013_043712_3ca44d08 | 0.923 | 0.006 | 0.009 | 0.033 | 0.308 | 0.356 | 0.000 | 0.567 |
| remote_loop_20261013_043816_dfd3e0a0 | 0.975 | 0.005 | 0.008 | 0.036 | 0.348 | 0.397 | 0.000 | 0.577 |
| remote_loop_20261013_044101_933eca37 | 0.892 | 0.006 | 0.010 | 0.027 | 0.299 | 0.342 | 0.000 | 0.550 |
| remote_loop_20261013_044141_7fb563b3 | 0.954 | 0.007 | 0.010 | 0.034 | 0.293 | 0.345 | 0.000 | 0.609 |
| remote_loop_20261013_044211_e719a9ca | 0.912 | 0.006 | 0.010 | 0.035 | 0.293 | 0.344 | 0.000 | 0.568 |
| remote_loop_20261013_044250_c01c5d9f | 0.936 | 0.008 | 0.010 | 0.027 | 0.306 | 0.352 | 0.000 | 0.584 |
| remote_loop_20261013_044525_7990fd2e | 0.973 | 0.007 | 0.008 | 0.040 | 0.312 | 0.367 | 0.000 | 0.606 |
| remote_loop_20261013_044641_06df011b | 0.944 | 0.006 | 0.009 | 0.036 | 0.322 | 0.373 | 0.000 | 0.572 |
| 20261012_040655_255759 | 1.261 | 0.006 | 0.008 | 0.059 | 0.325 | 0.399 | 0.000 | 0.863 |
| 20261012_043822_113279 | 1.347 | 0.007 | 0.020 | 0.134 | 0.337 | 0.497 | 0.000 | 0.849 |

Server total is inside request RTT, not added to it. RTT minus server total is residual network/tunnel/client overhead, not pure network latency. Receive-body time can itself include waiting for uploaded bytes. Processor and GPU transfer are measured together. Model prediction includes vision encoding, autoregressive action-token generation and decoding to seven values.

## Client request stages — averages in seconds

| Run | Encode (outside RTT) | Connect (outside RTT) | Send | Wait for headers | Read response |
| --- | --- | --- | --- | --- | --- |
| simulator_trial | — | — | — | — | — |
| remote_loop_20261010_060952_21db409c | 0.001 | 0.001 | 0.001 | 0.957 | 0.003 |
| remote_loop_20261010_072519_5186f692 | 0.001 | 0.001 | 0.001 | 0.910 | 0.000 |
| remote_loop_20261010_133609_315d8199 | 0.001 | 0.001 | 0.001 | 1.122 | 0.000 |
| remote_loop_20261011_003638_4fe3675c | 0.001 | 0.001 | 0.001 | 1.050 | 0.003 |
| remote_loop_20261011_021037_83d8c4a3 | 0.001 | 0.001 | 0.001 | 0.943 | 0.000 |
| remote_loop_20261011_023950_e87092ab | 0.001 | 0.001 | 0.001 | 1.356 | 0.000 |
| remote_loop_20261011_130030_90592976 | 0.001 | 0.001 | 0.001 | 0.904 | 0.000 |
| remote_loop_20261011_130522_3943abad | 0.001 | 0.001 | 0.001 | 0.919 | 0.000 |
| remote_loop_20261012_114419_a61c5550 | 0.001 | 0.001 | 0.001 | 1.059 | 0.000 |
| remote_loop_20261012_115022_efd46fc5 | 0.001 | 0.001 | 0.001 | 1.016 | 0.000 |
| remote_loop_20261012_115331_fcdda8c6 | 0.001 | 0.001 | 0.001 | 1.009 | 0.000 |
| remote_loop_20261012_122158_a28ecf48 | 0.001 | 0.001 | 0.001 | 0.994 | 0.000 |
| remote_loop_20261012_122230_1398de3f | 0.001 | 0.001 | 0.001 | 1.008 | 0.002 |
| remote_loop_20261012_122634_3ccc119e | 0.001 | 0.001 | 0.001 | 0.964 | 0.000 |
| remote_loop_20261012_122746_afdcf70c | 0.001 | 0.001 | 0.001 | 0.951 | 0.000 |
| remote_loop_20261012_122831_de4972a4 | 0.001 | 0.001 | 0.001 | 1.026 | 0.020 |
| remote_loop_20261013_004429_0ac8a961 | 0.001 | 0.004 | 0.001 | 1.266 | 0.000 |
| remote_loop_20261013_004457_d1ae30e2 | 0.001 | 0.004 | 0.000 | 1.154 | 0.000 |
| remote_loop_20261013_004516_7b5da887 | 0.001 | 0.003 | 0.000 | 1.175 | 0.000 |
| remote_loop_20261013_020816_48c2152f | 0.001 | 0.001 | 0.000 | 0.952 | 0.000 |
| remote_loop_20261013_020901_f64b721f | 0.001 | 0.001 | 0.001 | 0.985 | 0.000 |
| remote_loop_20261013_021001_2a2f89e2 | 0.001 | 0.001 | 0.001 | 1.021 | 0.000 |
| remote_loop_20261013_021421_598d0987 | 0.001 | 0.001 | 0.001 | 1.085 | 0.000 |
| remote_loop_20261013_021707_d8a114c5 | 0.001 | 0.001 | 0.001 | 1.075 | 0.000 |
| remote_loop_20261013_022040_8ed20480 | 0.001 | 0.001 | 0.001 | 1.020 | 0.000 |
| remote_loop_20261013_022319_a6d8b143 | 0.001 | 0.001 | 0.001 | 0.985 | 0.000 |
| remote_loop_20261013_022403_5dc12f75 | 0.001 | 0.001 | 0.001 | 0.992 | 0.000 |
| remote_loop_20261013_022854_75e43126 | 0.001 | 0.001 | 0.001 | 1.081 | 0.000 |
| remote_loop_20261013_023013_9609f73d | 0.000 | 0.001 | 0.001 | 1.111 | 0.000 |
| remote_loop_20261013_023115_96815de3 | 0.001 | 0.001 | 0.001 | 1.062 | 0.000 |
| remote_loop_20261013_024046_7faaeb28 | 0.001 | 0.001 | 0.001 | 1.050 | 0.000 |
| remote_loop_20261013_024249_f3af7673 | 0.001 | 0.001 | 0.001 | 1.065 | 0.000 |
| remote_loop_20261013_024522_2e573d8d | 0.001 | 0.001 | 0.001 | 1.047 | 0.000 |
| remote_loop_20261013_040917_acebfb07 | 0.001 | 0.001 | 0.001 | 0.887 | 0.000 |
| remote_loop_20261013_040954_239e0172 | 0.001 | 0.001 | 0.001 | 0.903 | 0.000 |
| remote_loop_20261013_041030_f12b9235 | 0.001 | 0.001 | 0.001 | 0.918 | 0.000 |
| remote_loop_20261013_041321_8805e94b | 0.001 | 0.001 | 0.001 | 0.990 | 0.000 |
| remote_loop_20261013_041412_8a739ca4 | 0.001 | 0.001 | 0.001 | 0.983 | 0.015 |
| remote_loop_20261013_041455_1cbdf9b8 | 0.001 | 0.001 | 0.001 | 0.937 | 0.000 |
| remote_loop_20261013_041734_b4164a41 | 0.001 | 0.001 | 0.001 | 1.006 | 0.000 |
| remote_loop_20261013_041813_f370b0c6 | 0.001 | 0.001 | 0.001 | 0.988 | 0.000 |
| remote_loop_20261013_041850_e86626e6 | 0.001 | 0.001 | 0.001 | 0.980 | 0.000 |
| remote_loop_20261013_042244_e5d85e46 | 0.001 | 0.001 | 0.001 | 0.949 | 0.000 |
| remote_loop_20261013_042322_289a8f5f | 0.001 | 0.001 | 0.001 | 0.932 | 0.000 |
| remote_loop_20261013_042402_f71b51e0 | 0.001 | 0.001 | 0.001 | 0.917 | 0.000 |
| remote_loop_20261013_042546_0a547490 | 0.001 | 0.001 | 0.001 | 0.952 | 0.000 |
| remote_loop_20261013_042618_7f5c0bb6 | 0.001 | 0.001 | 0.001 | 0.925 | 0.000 |
| remote_loop_20261013_042935_ac686ee3 | 0.001 | 0.001 | 0.001 | 0.964 | 0.000 |
| remote_loop_20261013_043008_c17a4b95 | 0.001 | 0.001 | 0.001 | 0.987 | 0.000 |
| remote_loop_20261013_043040_fa0cd102 | 0.001 | 0.001 | 0.001 | 1.071 | 0.000 |
| remote_loop_20261013_043307_12ea3445 | 0.001 | 0.001 | 0.001 | 0.927 | 0.000 |
| remote_loop_20261013_043343_b2368bf0 | 0.001 | 0.001 | 0.001 | 0.972 | 0.000 |
| remote_loop_20261013_043412_4c8ece6c | 0.001 | 0.001 | 0.001 | 0.933 | 0.000 |
| remote_loop_20261013_043611_4395c010 | 0.001 | 0.001 | 0.001 | 0.956 | 0.019 |
| remote_loop_20261013_043651_1d69c320 | — | — | — | — | — |
| remote_loop_20261013_043712_3ca44d08 | 0.001 | 0.001 | 0.001 | 0.922 | 0.000 |
| remote_loop_20261013_043816_dfd3e0a0 | 0.001 | 0.001 | 0.001 | 0.974 | 0.000 |
| remote_loop_20261013_044101_933eca37 | 0.001 | 0.001 | 0.001 | 0.892 | 0.000 |
| remote_loop_20261013_044141_7fb563b3 | 0.001 | 0.001 | 0.001 | 0.953 | 0.000 |
| remote_loop_20261013_044211_e719a9ca | 0.001 | 0.001 | 0.001 | 0.911 | 0.000 |
| remote_loop_20261013_044250_c01c5d9f | 0.001 | 0.001 | 0.001 | 0.935 | 0.000 |
| remote_loop_20261013_044525_7990fd2e | 0.001 | 0.001 | 0.001 | 0.972 | 0.000 |
| remote_loop_20261013_044641_06df011b | 0.001 | 0.001 | 0.001 | 0.943 | 0.000 |
| 20261012_040655_255759 | 0.002 | 0.001 | 0.001 | 1.260 | 0.000 |
| 20261012_043822_113279 | 0.002 | 0.001 | 0.001 | 1.346 | 0.000 |

Send, wait for response headers and read response are included in request RTT. Sending may buffer bytes locally; it is not a separately measured network upload duration. Waiting includes server processing and network delays.

## Local simulator breakdown — averages in seconds

| Run | Input capture/save | IK | Movement | Result capture/save | Local subtotal | Remaining cycle overhead |
| --- | --- | --- | --- | --- | --- | --- |
| simulator_trial | — | — | — | — | — | — |
| remote_loop_20261010_060952_21db409c | 0.022 | 0.001 | 0.446 | 0.025 | 0.494 | 0.010 |
| remote_loop_20261010_072519_5186f692 | 0.022 | 0.001 | 0.446 | 0.025 | 0.495 | 0.009 |
| remote_loop_20261010_133609_315d8199 | 0.023 | 0.001 | 0.446 | 0.027 | 0.497 | 0.010 |
| remote_loop_20261011_003638_4fe3675c | 0.022 | 0.001 | 0.446 | 0.029 | 0.498 | 0.010 |
| remote_loop_20261011_021037_83d8c4a3 | 0.023 | 0.001 | 0.445 | 0.028 | 0.497 | 0.010 |
| remote_loop_20261011_023950_e87092ab | 0.022 | 0.001 | 0.446 | 0.028 | 0.498 | 0.010 |
| remote_loop_20261011_130030_90592976 | 0.022 | 0.001 | 0.447 | 0.026 | 0.495 | 0.011 |
| remote_loop_20261011_130522_3943abad | 0.021 | 0.001 | 0.446 | 0.024 | 0.492 | 0.010 |
| remote_loop_20261012_114419_a61c5550 | 0.023 | 0.001 | 0.445 | 0.027 | 0.496 | 0.010 |
| remote_loop_20261012_115022_efd46fc5 | 0.024 | 0.001 | 0.445 | 0.026 | 0.496 | 0.011 |
| remote_loop_20261012_115331_fcdda8c6 | 0.025 | 0.001 | 0.445 | 0.024 | 0.495 | 0.011 |
| remote_loop_20261012_122158_a28ecf48 | 0.024 | 0.001 | 0.446 | 0.024 | 0.495 | 0.009 |
| remote_loop_20261012_122230_1398de3f | 0.022 | 0.001 | 0.446 | 0.025 | 0.495 | 0.010 |
| remote_loop_20261012_122634_3ccc119e | 0.024 | 0.001 | 0.433 | 0.024 | 0.482 | 0.009 |
| remote_loop_20261012_122746_afdcf70c | 0.023 | 0.001 | 0.442 | 0.023 | 0.489 | 0.011 |
| remote_loop_20261012_122831_de4972a4 | 0.024 | 0.001 | 0.447 | 0.025 | 0.497 | 0.009 |
| remote_loop_20261013_004429_0ac8a961 | 0.023 | 0.001 | 0.445 | 0.027 | 0.496 | 0.014 |
| remote_loop_20261013_004457_d1ae30e2 | 0.024 | 0.001 | 0.443 | 0.020 | 0.488 | 0.017 |
| remote_loop_20261013_004516_7b5da887 | 0.022 | 0.000 | 0.442 | 0.022 | 0.486 | 0.017 |
| remote_loop_20261013_020816_48c2152f | 0.021 | 0.006 | 0.379 | 0.023 | 0.429 | 0.013 |
| remote_loop_20261013_020901_f64b721f | 0.024 | 0.001 | 0.446 | 0.025 | 0.495 | 0.010 |
| remote_loop_20261013_021001_2a2f89e2 | 0.018 | 0.001 | 0.443 | 0.020 | 0.482 | 0.009 |
| remote_loop_20261013_021421_598d0987 | 0.023 | 0.001 | 0.446 | 0.025 | 0.495 | 0.010 |
| remote_loop_20261013_021707_d8a114c5 | 0.024 | 0.003 | 0.405 | 0.024 | 0.455 | 0.011 |
| remote_loop_20261013_022040_8ed20480 | 0.024 | 0.001 | 0.446 | 0.025 | 0.495 | 0.010 |
| remote_loop_20261013_022319_a6d8b143 | 0.023 | 0.001 | 0.446 | 0.024 | 0.493 | 0.010 |
| remote_loop_20261013_022403_5dc12f75 | 0.024 | 0.001 | 0.444 | 0.026 | 0.494 | 0.011 |
| remote_loop_20261013_022854_75e43126 | 0.025 | 0.001 | 0.445 | 0.025 | 0.496 | 0.009 |
| remote_loop_20261013_023013_9609f73d | 0.018 | 0.001 | 0.440 | 0.021 | 0.481 | 0.009 |
| remote_loop_20261013_023115_96815de3 | 0.023 | 0.001 | 0.446 | 0.025 | 0.495 | 0.010 |
| remote_loop_20261013_024046_7faaeb28 | 0.024 | 0.001 | 0.446 | 0.024 | 0.495 | 0.009 |
| remote_loop_20261013_024249_f3af7673 | 0.023 | 0.001 | 0.446 | 0.024 | 0.494 | 0.010 |
| remote_loop_20261013_024522_2e573d8d | 0.022 | 0.001 | 0.444 | 0.021 | 0.488 | 0.010 |
| remote_loop_20261013_040917_acebfb07 | 0.023 | 0.001 | 0.446 | 0.023 | 0.493 | 0.010 |
| remote_loop_20261013_040954_239e0172 | 0.024 | 0.001 | 0.447 | 0.023 | 0.495 | 0.011 |
| remote_loop_20261013_041030_f12b9235 | 0.023 | 0.001 | 0.446 | 0.022 | 0.492 | 0.011 |
| remote_loop_20261013_041321_8805e94b | 0.023 | 0.001 | 0.445 | 0.024 | 0.493 | 0.009 |
| remote_loop_20261013_041412_8a739ca4 | 0.023 | 0.001 | 0.447 | 0.023 | 0.494 | 0.011 |
| remote_loop_20261013_041455_1cbdf9b8 | 0.022 | 0.001 | 0.445 | 0.024 | 0.492 | 0.011 |
| remote_loop_20261013_041734_b4164a41 | 0.023 | 0.001 | 0.446 | 0.025 | 0.495 | 0.009 |
| remote_loop_20261013_041813_f370b0c6 | 0.023 | 0.001 | 0.444 | 0.024 | 0.492 | 0.012 |
| remote_loop_20261013_041850_e86626e6 | 0.024 | 0.001 | 0.447 | 0.023 | 0.495 | 0.010 |
| remote_loop_20261013_042244_e5d85e46 | 0.023 | 0.001 | 0.447 | 0.024 | 0.495 | 0.009 |
| remote_loop_20261013_042322_289a8f5f | 0.023 | 0.001 | 0.446 | 0.023 | 0.493 | 0.009 |
| remote_loop_20261013_042402_f71b51e0 | 0.023 | 0.001 | 0.447 | 0.021 | 0.492 | 0.012 |
| remote_loop_20261013_042546_0a547490 | 0.022 | 0.001 | 0.447 | 0.023 | 0.493 | 0.012 |
| remote_loop_20261013_042618_7f5c0bb6 | 0.022 | 0.001 | 0.444 | 0.023 | 0.490 | 0.010 |
| remote_loop_20261013_042935_ac686ee3 | 0.023 | 0.001 | 0.444 | 0.022 | 0.490 | 0.009 |
| remote_loop_20261013_043008_c17a4b95 | 0.022 | 0.001 | 0.445 | 0.022 | 0.490 | 0.008 |
| remote_loop_20261013_043040_fa0cd102 | 0.023 | 0.001 | 0.447 | 0.021 | 0.491 | 0.009 |
| remote_loop_20261013_043307_12ea3445 | 0.022 | 0.001 | 0.447 | 0.023 | 0.493 | 0.009 |
| remote_loop_20261013_043343_b2368bf0 | 0.022 | 0.001 | 0.446 | 0.023 | 0.492 | 0.011 |
| remote_loop_20261013_043412_4c8ece6c | 0.022 | 0.001 | 0.444 | 0.023 | 0.490 | 0.010 |
| remote_loop_20261013_043611_4395c010 | 0.023 | 0.001 | 0.444 | 0.023 | 0.491 | 0.010 |
| remote_loop_20261013_043651_1d69c320 | — | — | — | — | — | — |
| remote_loop_20261013_043712_3ca44d08 | 0.023 | 0.001 | 0.446 | 0.024 | 0.495 | 0.011 |
| remote_loop_20261013_043816_dfd3e0a0 | 0.023 | 0.001 | 0.445 | 0.022 | 0.491 | 0.011 |
| remote_loop_20261013_044101_933eca37 | 0.023 | 0.001 | 0.447 | 0.024 | 0.495 | 0.009 |
| remote_loop_20261013_044141_7fb563b3 | 0.023 | 0.001 | 0.447 | 0.022 | 0.492 | 0.009 |
| remote_loop_20261013_044211_e719a9ca | 0.022 | 0.001 | 0.443 | 0.021 | 0.487 | 0.008 |
| remote_loop_20261013_044250_c01c5d9f | 0.022 | 0.001 | 0.447 | 0.023 | 0.493 | 0.012 |
| remote_loop_20261013_044525_7990fd2e | 0.022 | 0.001 | 0.448 | 0.023 | 0.494 | 0.008 |
| remote_loop_20261013_044641_06df011b | 0.023 | 0.001 | 0.448 | 0.023 | 0.495 | 0.010 |
| 20261012_040655_255759 | — | — | — | — | — | — |
| 20261012_043822_113279 | — | — | — | — | — | — |

Full cycle = request RTT + local subtotal + remaining cycle overhead, using records with all required fields. Remaining overhead includes client encoding/connection work and recording outside named stages. Movement may be deliberately paced by the live viewer. Pauses or interruptions within a recorded cycle can affect timing. Each column averages available measurements only; different coverage can prevent exact subtotal agreement.

## Setup and comparison context

| Run | VRAM GiB | ECC | Model / checkpoint | Load s | Preparation s | Storage | PyTorch / Transformers | Precision / attention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simulator_trial | 44.432 | Not recorded | /workspace/openvla-7b | 63.043 | 0.055 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261010_060952_21db409c | 44.432 | Not recorded | /workspace/openvla-7b | 64.261 | 0.049 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261010_072519_5186f692 | 44.432 | Not recorded | /workspace/openvla-7b | 41.930 | 0.048 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261010_133609_315d8199 | 44.424 | Not recorded | /workspace/panda-training/runs/lora_20261010_130927_064468/checkpoint_000020 | 5.265 | 0.049 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261011_003638_4fe3675c | 47.402 | Not recorded | /workspace/panda-training/sessions/panda-train-20261010-233038-932/runs/training/checkpoint_000500 | 5.498 | 0.048 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261011_021037_83d8c4a3 | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261010-233038-932/runs/training/checkpoint_001000 | 6.027 | 0.050 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261011_023950_e87092ab | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261010-233038-932/runs/training/checkpoint_001000 | 5.844 | 0.047 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261011_130030_90592976 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261011-090041-1584/runs/training/checkpoint_003000 | 5.384 | 0.045 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261011_130522_3943abad | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261011-090041-1584/runs/training/checkpoint_005000 | 5.866 | 0.038 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261012_114419_a61c5550 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261012-072437-856/runs/training/checkpoint_002000 | 6.745 | 0.057 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261012_115022_efd46fc5 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261012-072437-856/runs/training/checkpoint_002000 | 6.572 | 0.056 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261012_115331_fcdda8c6 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261012-072437-856/runs/training/checkpoint_002000 | 6.775 | 0.052 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261012_122158_a28ecf48 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261012-072437-856/runs/training/checkpoint_002000 | 5.623 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261012_122230_1398de3f | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261012-072437-856/runs/training/checkpoint_002000 | 5.623 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261012_122634_3ccc119e | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261012-072437-856/runs/training/checkpoint_002000 | 5.623 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261012_122746_afdcf70c | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261012-072437-856/runs/training/checkpoint_002000 | 5.623 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261012_122831_de4972a4 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261012-072437-856/runs/training/checkpoint_002000 | 5.623 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_004429_0ac8a961 | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261012-072437-856/runs/training/checkpoint_005000 | 7.165 | 0.056 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_004457_d1ae30e2 | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261012-072437-856/runs/training/checkpoint_005000 | 7.165 | 0.056 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_004516_7b5da887 | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261012-072437-856/runs/training/checkpoint_005000 | 7.165 | 0.056 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_020816_48c2152f | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-013939-2113/runs/training/checkpoint_000400 | 6.666 | 0.048 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_020901_f64b721f | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-013939-2113/runs/training/checkpoint_000400 | 6.666 | 0.048 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_021001_2a2f89e2 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-013939-2113/runs/training/checkpoint_000400 | 6.666 | 0.048 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_021421_598d0987 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-013939-2113/runs/training/checkpoint_000000 | 9.427 | 0.038 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_021707_d8a114c5 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-013939-2113/runs/training/checkpoint_000000 | 9.427 | 0.038 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_022040_8ed20480 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-013939-2113/runs/training/checkpoint_000050 | 8.518 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_022319_a6d8b143 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-013939-2113/runs/training/checkpoint_000050 | 8.518 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_022403_5dc12f75 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-013939-2113/runs/training/checkpoint_000050 | 8.518 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_022854_75e43126 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-013939-2113/runs/training/checkpoint_000100 | 7.146 | 0.037 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_023013_9609f73d | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-013939-2113/runs/training/checkpoint_000100 | 7.146 | 0.037 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_023115_96815de3 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-013939-2113/runs/training/checkpoint_000100 | 7.146 | 0.037 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_024046_7faaeb28 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-013939-2113/runs/training/checkpoint_000200 | 5.608 | 0.039 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_024249_f3af7673 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-013939-2113/runs/training/checkpoint_000200 | 5.608 | 0.039 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_024522_2e573d8d | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-013939-2113/runs/training/checkpoint_000200 | 5.608 | 0.039 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_040917_acebfb07 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000000 | 5.801 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_040954_239e0172 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000000 | 5.801 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_041030_f12b9235 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000000 | 5.801 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_041321_8805e94b | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000010 | 5.869 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_041412_8a739ca4 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000010 | 5.869 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_041455_1cbdf9b8 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000010 | 5.869 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_041734_b4164a41 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.537 | 0.039 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_041813_f370b0c6 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.537 | 0.039 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_041850_e86626e6 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.537 | 0.039 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_042244_e5d85e46 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.537 | 0.039 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_042322_289a8f5f | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.537 | 0.039 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_042402_f71b51e0 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.537 | 0.039 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_042546_0a547490 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.537 | 0.039 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_042618_7f5c0bb6 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.537 | 0.039 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_042935_ac686ee3 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.076 | 0.072 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_043008_c17a4b95 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.076 | 0.072 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_043040_fa0cd102 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.076 | 0.072 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_043307_12ea3445 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.834 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_043343_b2368bf0 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.834 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_043412_4c8ece6c | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.834 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_043611_4395c010 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.834 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_043651_1d69c320 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.834 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_043712_3ca44d08 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.834 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_043816_dfd3e0a0 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.834 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_044101_933eca37 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 5.687 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_044141_7fb563b3 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 5.687 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_044211_e719a9ca | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 5.687 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_044250_c01c5d9f | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 5.687 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_044525_7990fd2e | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 6.411 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261013_044641_06df011b | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 6.411 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| 20261012_040655_255759 | 47.406 | Disabled | /workspace/openvla-7b | 52.546 | — | Not recorded | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| 20261012_043822_113279 | 23.547 | Disabled | /workspace/openvla-7b | 46.298 | — | Not recorded | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |

Model loading includes processor setup, checkpoint reading, construction and GPU transfer; it does not isolate storage-to-VRAM time. Historical ECC is not inferred from unrelated training pods.

## Refreshing this report

New local simulator trials refresh this report automatically after saving their run report. No cloud connections or predictions are made by report generation.

```bash
.venv/bin/python -m analysis.summarize_gpu_inference
```

Hourly rates are stored in `inference/gpu_inference_rates.json`. Raw measurements remain in `outputs/<run>/startup_info.json` and `step_*/execution.json`. Each run also keeps `RUN_REPORT.md`.
