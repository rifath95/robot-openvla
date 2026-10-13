# GPU inference comparison

One request = one RGB image + instruction → one seven-value action. Inference batch size is 1.

Timings are seconds. The first 5 recorded requests per trial are excluded as warmup. Short trials without remaining samples show missing timings. These are observed simulator runs, not identical-input hardware benchmarks; model/checkpoint, instruction, network, Mac, storage and software differences can affect results. Rejected/held actions remain prediction requests. Combined benchmarks start with 5 warmup requests and 20 measured inference requests, both on the cloud and from the Mac when connected. Cloud-loopback RTT is never reported as Mac latency. When Mac measurements exist, the server breakdown uses those same responses to keep RTT and server timing paired; otherwise it uses the cloud samples. Both raw sets are retained.


## Average by GPU type

Means are weighted by measured request counts, not an average of rounded table values. There is one row per reported GPU type: all A40 runs are combined, for example. These overall averages mix VRAM/ECC configurations, model checkpoints, software versions, inputs, and simulator/benchmark runs; use the detailed tables below to inspect those differences. They describe our observed runs rather than an isolated hardware ranking. Local timing is Mac-dependent. Each metric includes only runs that recorded it; cloud-only runs contribute server timing but not Mac RTT or simulator time. The per-run table below preserves each observation.

| GPU | Runs | Measured requests | Mean cycle s | Mean request RTT s | Mean prediction s | USD/hour | 100 requests USD | 100 simulator cycles USD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVIDIA A40 | 255 | 5162 | 1.400 | 1.134 | 0.335 | 0.59 | 0.0186 | 0.0230 |
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
| NVIDIA A40 | checkpoint_000050_seed_1000_blue | 19/24 | 1.320 | 1.231 | 1.364 | 0.342 | 0.59 | 132.0 | 0.0216 | 0.0228 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1000_red | 13/18 | 1.408 | 1.321 | 1.500 | 0.363 | 0.59 | 140.8 | 0.0231 | 0.0243 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1001_blue | 14/19 | 1.256 | 1.161 | 1.289 | 0.338 | 0.59 | 125.6 | 0.0206 | 0.0218 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1001_red | 16/21 | 1.287 | 1.198 | 1.429 | 0.320 | 0.59 | 128.7 | 0.0211 | 0.0223 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1002_blue | 18/23 | 1.352 | 1.257 | 1.545 | 0.342 | 0.59 | 135.2 | 0.0222 | 0.0234 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1002_red | 18/23 | 1.303 | 1.206 | 1.291 | 0.333 | 0.59 | 130.3 | 0.0213 | 0.0225 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1003_blue | 20/25 | 1.427 | 1.337 | 1.760 | 0.334 | 0.59 | 142.7 | 0.0234 | 0.0246 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1003_red | 18/23 | 1.292 | 1.201 | 1.343 | 0.324 | 0.59 | 129.2 | 0.0212 | 0.0224 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1004_blue | 19/24 | 1.439 | 1.346 | 1.815 | 0.338 | 0.59 | 143.9 | 0.0236 | 0.0248 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1004_red | 20/25 | 1.354 | 1.261 | 1.453 | 0.333 | 0.59 | 135.4 | 0.0222 | 0.0234 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1005_blue | 19/24 | 1.324 | 1.226 | 1.375 | 0.353 | 0.59 | 132.4 | 0.0217 | 0.0229 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1005_red | 18/23 | 1.506 | 1.414 | 1.854 | 0.346 | 0.59 | 150.6 | 0.0247 | 0.0259 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1006_blue | 14/19 | 1.335 | 1.242 | 1.423 | 0.329 | 0.59 | 133.5 | 0.0219 | 0.0231 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1006_red | 20/25 | 1.339 | 1.251 | 1.382 | 0.343 | 0.59 | 133.9 | 0.0219 | 0.0231 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1007_blue | 12/17 | 1.574 | 1.497 | 2.654 | 0.325 | 0.59 | 157.4 | 0.0258 | 0.0270 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1007_red | 14/19 | 1.572 | 1.475 | 2.125 | 0.343 | 0.59 | 157.2 | 0.0258 | 0.0270 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1008_blue | 18/23 | 1.527 | 1.439 | 1.773 | 0.342 | 0.59 | 152.7 | 0.0250 | 0.0262 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1008_red | 19/24 | 1.319 | 1.229 | 1.360 | 0.348 | 0.59 | 131.9 | 0.0216 | 0.0228 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1009_blue | 15/20 | 1.393 | 1.304 | 1.481 | 0.345 | 0.59 | 139.3 | 0.0228 | 0.0240 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1009_red | 12/17 | 1.412 | 1.317 | 1.554 | 0.332 | 0.59 | 141.2 | 0.0231 | 0.0243 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1000_blue | 14/19 | 1.413 | 1.339 | 1.590 | 0.328 | 0.59 | 141.3 | 0.0232 | 0.0244 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1000_red | 12/17 | 1.434 | 1.343 | 1.609 | 0.325 | 0.59 | 143.4 | 0.0235 | 0.0247 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1001_blue | 14/19 | 1.433 | 1.363 | 1.738 | 0.323 | 0.59 | 143.3 | 0.0235 | 0.0247 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1001_red | 16/21 | 1.675 | 1.596 | 1.827 | 0.344 | 0.59 | 167.5 | 0.0274 | 0.0286 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1002_blue | 14/19 | 1.328 | 1.242 | 1.443 | 0.342 | 0.59 | 132.8 | 0.0218 | 0.0230 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1002_red | 10/15 | 1.304 | 1.217 | 1.372 | 0.332 | 0.59 | 130.4 | 0.0214 | 0.0226 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1003_blue | 18/23 | 1.442 | 1.350 | 1.805 | 0.347 | 0.59 | 144.2 | 0.0236 | 0.0248 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1003_red | 17/22 | 1.404 | 1.323 | 1.612 | 0.335 | 0.59 | 140.4 | 0.0230 | 0.0242 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1004_blue | 20/25 | 1.360 | 1.272 | 1.513 | 0.334 | 0.59 | 136.0 | 0.0223 | 0.0235 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1004_red | 15/20 | 1.378 | 1.287 | 1.628 | 0.357 | 0.59 | 137.8 | 0.0226 | 0.0238 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1005_blue | 14/19 | 1.329 | 1.243 | 1.417 | 0.361 | 0.59 | 132.9 | 0.0218 | 0.0230 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1005_red | 17/22 | 1.367 | 1.283 | 1.547 | 0.346 | 0.59 | 136.7 | 0.0224 | 0.0236 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1006_blue | 13/18 | 1.311 | 1.227 | 1.370 | 0.335 | 0.59 | 131.1 | 0.0215 | 0.0227 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1006_red | 18/23 | 1.431 | 1.336 | 1.717 | 0.352 | 0.59 | 143.1 | 0.0234 | 0.0246 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1007_blue | 9/14 | 1.379 | 1.292 | 1.462 | 0.361 | 0.59 | 137.9 | 0.0226 | 0.0238 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1007_red | 16/21 | 1.349 | 1.261 | 1.457 | 0.320 | 0.59 | 134.9 | 0.0221 | 0.0233 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1008_blue | 16/21 | 1.384 | 1.298 | 1.423 | 0.350 | 0.59 | 138.4 | 0.0227 | 0.0239 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1008_red | 14/19 | 1.381 | 1.292 | 1.350 | 0.358 | 0.59 | 138.1 | 0.0226 | 0.0238 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1009_blue | 15/20 | 1.344 | 1.261 | 1.561 | 0.339 | 0.59 | 134.4 | 0.0220 | 0.0232 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1009_red | 12/17 | 1.348 | 1.263 | 1.421 | 0.352 | 0.59 | 134.8 | 0.0221 | 0.0233 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1000_blue | 14/19 | 1.344 | 1.261 | 1.377 | 0.355 | 0.59 | 134.4 | 0.0220 | 0.0232 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1000_red | 12/17 | 1.506 | 1.426 | 1.683 | 0.377 | 0.59 | 150.6 | 0.0247 | 0.0258 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1001_blue | 15/20 | 1.316 | 1.228 | 1.418 | 0.353 | 0.59 | 131.6 | 0.0216 | 0.0227 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1001_red | 22/27 | 1.331 | 1.239 | 1.438 | 0.332 | 0.59 | 133.1 | 0.0218 | 0.0229 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1002_blue | 14/19 | 1.349 | 1.254 | 1.491 | 0.347 | 0.59 | 134.9 | 0.0221 | 0.0232 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1002_red | 10/15 | 1.365 | 1.274 | 1.469 | 0.349 | 0.59 | 136.5 | 0.0224 | 0.0235 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1003_blue | 12/17 | 1.309 | 1.218 | 1.394 | 0.358 | 0.59 | 130.9 | 0.0215 | 0.0226 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1003_red | 14/19 | 1.367 | 1.273 | 1.500 | 0.349 | 0.59 | 136.7 | 0.0224 | 0.0235 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1004_blue | 20/25 | 1.385 | 1.291 | 1.430 | 0.354 | 0.59 | 138.5 | 0.0227 | 0.0238 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1004_red | 21/26 | 1.360 | 1.267 | 1.413 | 0.357 | 0.59 | 136.0 | 0.0223 | 0.0234 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1005_blue | 14/19 | 1.608 | 1.520 | 1.815 | 0.356 | 0.59 | 160.8 | 0.0264 | 0.0275 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1005_red | 12/17 | 1.327 | 1.236 | 1.404 | 0.325 | 0.59 | 132.7 | 0.0217 | 0.0229 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1006_blue | 14/19 | 1.328 | 1.241 | 1.409 | 0.328 | 0.59 | 132.8 | 0.0218 | 0.0229 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1006_red | 14/19 | 1.391 | 1.318 | 1.570 | 0.360 | 0.59 | 139.1 | 0.0228 | 0.0239 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1007_blue | 9/14 | 1.343 | 1.258 | 1.492 | 0.338 | 0.59 | 134.3 | 0.0220 | 0.0231 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1007_red | 20/25 | 1.368 | 1.278 | 1.409 | 0.349 | 0.59 | 136.8 | 0.0224 | 0.0235 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1008_blue | 18/23 | 1.327 | 1.239 | 1.504 | 0.342 | 0.59 | 132.7 | 0.0217 | 0.0229 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1008_red | 14/19 | 1.369 | 1.284 | 1.539 | 0.365 | 0.59 | 136.9 | 0.0224 | 0.0236 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1009_blue | 16/21 | 1.344 | 1.256 | 1.488 | 0.361 | 0.59 | 134.4 | 0.0220 | 0.0231 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1009_red | 17/22 | 1.305 | 1.218 | 1.434 | 0.337 | 0.59 | 130.5 | 0.0214 | 0.0225 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1000_blue | 12/17 | 1.403 | 1.318 | 1.426 | 0.334 | 0.59 | 140.3 | 0.0230 | 0.0240 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1000_red | 12/17 | 1.430 | 1.343 | 1.664 | 0.345 | 0.59 | 143.0 | 0.0234 | 0.0245 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1001_blue | 14/19 | 1.339 | 1.273 | 1.480 | 0.345 | 0.59 | 133.9 | 0.0220 | 0.0230 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1001_red | 16/21 | 1.331 | 1.251 | 1.431 | 0.329 | 0.59 | 133.1 | 0.0218 | 0.0229 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1002_blue | 18/23 | 1.292 | 1.203 | 1.298 | 0.325 | 0.59 | 129.2 | 0.0212 | 0.0222 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1002_red | 10/15 | 1.289 | 1.198 | 1.258 | 0.339 | 0.59 | 128.9 | 0.0211 | 0.0222 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1003_blue | 12/17 | 1.400 | 1.309 | 1.649 | 0.339 | 0.59 | 140.0 | 0.0229 | 0.0240 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1003_red | 14/19 | 1.322 | 1.234 | 1.452 | 0.327 | 0.59 | 132.2 | 0.0217 | 0.0227 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1004_blue | 19/24 | 1.285 | 1.195 | 1.316 | 0.329 | 0.59 | 128.5 | 0.0211 | 0.0221 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1004_red | 15/20 | 1.391 | 1.302 | 1.753 | 0.332 | 0.59 | 139.1 | 0.0228 | 0.0238 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1005_blue | 14/19 | 1.309 | 1.227 | 1.330 | 0.352 | 0.59 | 130.9 | 0.0215 | 0.0225 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1005_red | 12/17 | 1.352 | 1.261 | 1.433 | 0.320 | 0.59 | 135.2 | 0.0222 | 0.0232 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1006_blue | 14/19 | 1.279 | 1.194 | 1.307 | 0.347 | 0.59 | 127.9 | 0.0210 | 0.0220 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1006_red | 18/23 | 1.349 | 1.263 | 1.773 | 0.329 | 0.59 | 134.9 | 0.0221 | 0.0231 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1007_blue | 9/14 | 1.283 | 1.197 | 1.308 | 0.335 | 0.59 | 128.3 | 0.0210 | 0.0221 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1007_red | 20/25 | 1.298 | 1.213 | 1.392 | 0.336 | 0.59 | 129.8 | 0.0213 | 0.0223 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1008_blue | 19/24 | 1.298 | 1.214 | 1.366 | 0.345 | 0.59 | 129.8 | 0.0213 | 0.0223 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1008_red | 14/19 | 1.329 | 1.248 | 1.352 | 0.347 | 0.59 | 132.9 | 0.0218 | 0.0228 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1009_blue | 15/20 | 1.322 | 1.238 | 1.370 | 0.346 | 0.59 | 132.2 | 0.0217 | 0.0227 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1009_red | 18/23 | 1.298 | 1.209 | 1.384 | 0.362 | 0.59 | 129.8 | 0.0213 | 0.0223 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1000_black-purple_black | 15/20 | 1.293 | 1.208 | 1.344 | 0.366 | 0.59 | 129.3 | 0.0212 | 0.0224 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1000_black-purple_purple | 12/17 | 1.252 | 1.161 | 1.321 | 0.352 | 0.59 | 125.2 | 0.0205 | 0.0217 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1000_green-yellow_green | 18/23 | 1.202 | 1.143 | 1.255 | 0.329 | 0.59 | 120.2 | 0.0197 | 0.0209 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1000_green-yellow_yellow | 18/23 | 1.242 | 1.180 | 1.292 | 0.340 | 0.59 | 124.2 | 0.0204 | 0.0216 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1001_black-purple_black | 14/19 | 1.244 | 1.160 | 1.303 | 0.350 | 0.59 | 124.4 | 0.0204 | 0.0216 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1001_black-purple_purple | 15/20 | 1.327 | 1.242 | 1.436 | 0.367 | 0.59 | 132.7 | 0.0218 | 0.0230 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1001_green-yellow_green | 21/26 | 1.274 | 1.210 | 1.377 | 0.349 | 0.59 | 127.4 | 0.0209 | 0.0221 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1001_green-yellow_yellow | 14/19 | 1.257 | 1.197 | 1.297 | 0.352 | 0.59 | 125.7 | 0.0206 | 0.0218 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1002_black-purple_black | 11/16 | 1.329 | 1.251 | 1.590 | 0.351 | 0.59 | 132.9 | 0.0218 | 0.0230 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1002_black-purple_purple | 12/17 | 1.239 | 1.158 | 1.255 | 0.344 | 0.59 | 123.9 | 0.0203 | 0.0215 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1002_green-yellow_green | 19/24 | 1.286 | 1.223 | 1.470 | 0.338 | 0.59 | 128.6 | 0.0211 | 0.0223 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1002_green-yellow_yellow | 19/24 | 1.255 | 1.193 | 1.406 | 0.333 | 0.59 | 125.5 | 0.0206 | 0.0218 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1003_black-purple_black | 19/24 | 1.245 | 1.150 | 1.251 | 0.351 | 0.59 | 124.5 | 0.0204 | 0.0216 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1003_black-purple_purple | 20/25 | 1.255 | 1.178 | 1.304 | 0.360 | 0.59 | 125.5 | 0.0206 | 0.0218 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1003_green-yellow_green | 17/22 | 1.293 | 1.218 | 1.453 | 0.343 | 0.59 | 129.3 | 0.0212 | 0.0224 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1003_green-yellow_yellow | 19/24 | 1.232 | 1.152 | 1.457 | 0.326 | 0.59 | 123.2 | 0.0202 | 0.0214 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1004_black-purple_black | 20/25 | 1.273 | 1.183 | 1.339 | 0.366 | 0.59 | 127.3 | 0.0209 | 0.0221 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1004_black-purple_purple | 19/24 | 1.234 | 1.148 | 1.316 | 0.352 | 0.59 | 123.4 | 0.0202 | 0.0214 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1004_green-yellow_green | 20/25 | 1.241 | 1.162 | 1.419 | 0.334 | 0.59 | 124.1 | 0.0203 | 0.0216 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1004_green-yellow_yellow | 21/26 | 1.245 | 1.178 | 1.342 | 0.329 | 0.59 | 124.5 | 0.0204 | 0.0216 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1005_black-purple_black | 18/23 | 1.224 | 1.136 | 1.285 | 0.346 | 0.59 | 122.4 | 0.0201 | 0.0213 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1005_black-purple_purple | 18/23 | 1.251 | 1.156 | 1.314 | 0.360 | 0.59 | 125.1 | 0.0205 | 0.0217 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1006_black-purple_black | 18/23 | 1.572 | 1.487 | 2.640 | 0.324 | 0.59 | 157.2 | 0.0258 | 0.0270 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1006_black-purple_purple | 20/25 | 1.219 | 1.133 | 1.236 | 0.326 | 0.59 | 121.9 | 0.0200 | 0.0212 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1007_black-purple_black | 11/16 | 1.262 | 1.177 | 1.376 | 0.335 | 0.59 | 126.2 | 0.0207 | 0.0219 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1007_black-purple_purple | 19/24 | 1.251 | 1.163 | 1.333 | 0.324 | 0.59 | 125.1 | 0.0205 | 0.0217 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1008_black-purple_black | 22/27 | 1.262 | 1.176 | 1.273 | 0.360 | 0.59 | 126.2 | 0.0207 | 0.0219 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1008_black-purple_purple | 18/23 | 1.232 | 1.154 | 1.294 | 0.338 | 0.59 | 123.2 | 0.0202 | 0.0214 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1009_black-purple_black | 21/26 | 1.245 | 1.184 | 1.346 | 0.344 | 0.59 | 124.5 | 0.0204 | 0.0216 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1009_black-purple_purple | 11/16 | 1.261 | 1.200 | 1.386 | 0.344 | 0.59 | 126.1 | 0.0207 | 0.0219 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1000 | 12/17 | 1.194 | 1.109 | 1.184 | 0.318 | 0.59 | 119.4 | 0.0196 | 0.0205 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1001 | 22/27 | 1.272 | 1.190 | 1.328 | 0.325 | 0.59 | 127.2 | 0.0208 | 0.0218 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1002 | 12/17 | 1.267 | 1.180 | 1.373 | 0.321 | 0.59 | 126.7 | 0.0208 | 0.0217 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1003 | 18/23 | 1.277 | 1.190 | 1.385 | 0.334 | 0.59 | 127.7 | 0.0209 | 0.0219 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1004 | 15/20 | 1.241 | 1.155 | 1.277 | 0.345 | 0.59 | 124.1 | 0.0203 | 0.0213 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1005 | 12/17 | 1.269 | 1.184 | 1.450 | 0.316 | 0.59 | 126.9 | 0.0208 | 0.0218 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1006 | 12/17 | 1.265 | 1.185 | 1.340 | 0.353 | 0.59 | 126.5 | 0.0207 | 0.0217 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1007 | 20/25 | 1.263 | 1.180 | 1.366 | 0.319 | 0.59 | 126.3 | 0.0207 | 0.0217 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1008 | 17/22 | 1.204 | 1.120 | 1.209 | 0.326 | 0.59 | 120.4 | 0.0197 | 0.0207 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1009 | 12/17 | 1.234 | 1.148 | 1.259 | 0.326 | 0.59 | 123.4 | 0.0202 | 0.0212 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1010 | 19/24 | 1.235 | 1.149 | 1.403 | 0.322 | 0.59 | 123.5 | 0.0202 | 0.0212 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1011 | 13/18 | 1.243 | 1.158 | 1.281 | 0.319 | 0.59 | 124.3 | 0.0204 | 0.0213 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1012 | 17/22 | 1.235 | 1.151 | 1.300 | 0.326 | 0.59 | 123.5 | 0.0202 | 0.0212 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1013 | 16/21 | 1.176 | 1.093 | 1.169 | 0.304 | 0.59 | 117.6 | 0.0193 | 0.0202 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1014 | 21/26 | 1.221 | 1.135 | 1.207 | 0.334 | 0.59 | 122.1 | 0.0200 | 0.0210 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1015 | 18/23 | 1.189 | 1.105 | 1.179 | 0.334 | 0.59 | 118.9 | 0.0195 | 0.0204 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1016 | 20/25 | 1.198 | 1.113 | 1.309 | 0.328 | 0.59 | 119.8 | 0.0196 | 0.0206 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1017 | 13/18 | 1.219 | 1.134 | 1.229 | 0.346 | 0.59 | 121.9 | 0.0200 | 0.0209 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1018 | 19/24 | 1.184 | 1.101 | 1.187 | 0.328 | 0.59 | 118.4 | 0.0194 | 0.0204 | Simulator |
| NVIDIA A40 | checkpoint_000050_seed_1019 | 20/25 | 1.201 | 1.121 | 1.237 | 0.331 | 0.59 | 120.1 | 0.0197 | 0.0206 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1000 | 15/20 | 1.601 | 1.516 | 1.821 | 0.342 | 0.59 | 160.1 | 0.0262 | 0.0272 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1001 | 20/25 | 1.287 | 1.208 | 1.385 | 0.342 | 0.59 | 128.7 | 0.0211 | 0.0221 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1002 | 10/15 | 1.302 | 1.218 | 1.576 | 0.346 | 0.59 | 130.2 | 0.0213 | 0.0224 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1003 | 17/22 | 1.300 | 1.218 | 1.421 | 0.340 | 0.59 | 130.0 | 0.0213 | 0.0223 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1004 | 15/20 | 1.309 | 1.228 | 1.502 | 0.361 | 0.59 | 130.9 | 0.0215 | 0.0225 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1005 | 12/17 | 1.338 | 1.253 | 1.742 | 0.344 | 0.59 | 133.8 | 0.0219 | 0.0229 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1006 | 11/16 | 1.275 | 1.193 | 1.418 | 0.331 | 0.59 | 127.5 | 0.0209 | 0.0219 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1007 | 14/19 | 1.248 | 1.164 | 1.387 | 0.341 | 0.59 | 124.8 | 0.0204 | 0.0215 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1008 | 16/21 | 1.223 | 1.138 | 1.237 | 0.356 | 0.59 | 122.3 | 0.0200 | 0.0211 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1009 | 11/16 | 1.231 | 1.145 | 1.283 | 0.324 | 0.59 | 123.1 | 0.0202 | 0.0212 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1010 | 15/20 | 1.299 | 1.214 | 1.427 | 0.348 | 0.59 | 129.9 | 0.0213 | 0.0223 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1011 | 11/16 | 1.209 | 1.125 | 1.230 | 0.329 | 0.59 | 120.9 | 0.0198 | 0.0208 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1012 | 10/15 | 1.255 | 1.170 | 1.435 | 0.341 | 0.59 | 125.5 | 0.0206 | 0.0216 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1013 | 14/19 | 1.306 | 1.219 | 1.413 | 0.349 | 0.59 | 130.6 | 0.0214 | 0.0224 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1014 | 14/19 | 1.289 | 1.208 | 1.444 | 0.329 | 0.59 | 128.9 | 0.0211 | 0.0221 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1015 | 17/22 | 1.328 | 1.244 | 1.541 | 0.353 | 0.59 | 132.8 | 0.0218 | 0.0228 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1016 | 20/25 | 1.217 | 1.131 | 1.215 | 0.326 | 0.59 | 121.7 | 0.0199 | 0.0210 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1017 | 13/18 | 2.139 | 2.048 | 4.728 | 0.331 | 0.59 | 213.9 | 0.0351 | 0.0361 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1018 | 11/16 | 1.309 | 1.226 | 1.393 | 0.347 | 0.59 | 130.9 | 0.0215 | 0.0225 | Simulator |
| NVIDIA A40 | checkpoint_000100_seed_1019 | 13/18 | 1.476 | 1.390 | 1.999 | 0.348 | 0.59 | 147.6 | 0.0242 | 0.0252 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1000 | 10/15 | 1.270 | 1.188 | 1.358 | 0.359 | 0.59 | 127.0 | 0.0208 | 0.0217 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1001 | 20/25 | 1.227 | 1.144 | 1.230 | 0.347 | 0.59 | 122.7 | 0.0201 | 0.0209 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1002 | 10/15 | 1.188 | 1.104 | 1.151 | 0.331 | 0.59 | 118.8 | 0.0195 | 0.0203 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1003 | 17/22 | 1.265 | 1.181 | 1.362 | 0.333 | 0.59 | 126.5 | 0.0207 | 0.0216 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1004 | 15/20 | 1.266 | 1.182 | 1.419 | 0.342 | 0.59 | 126.6 | 0.0208 | 0.0216 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1005 | 12/17 | 1.287 | 1.198 | 1.373 | 0.363 | 0.59 | 128.7 | 0.0211 | 0.0219 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1006 | 11/16 | 1.290 | 1.193 | 1.347 | 0.346 | 0.59 | 129.0 | 0.0211 | 0.0220 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1007 | 15/20 | 1.269 | 1.176 | 1.316 | 0.355 | 0.59 | 126.9 | 0.0208 | 0.0216 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1008 | 17/22 | 1.251 | 1.157 | 1.254 | 0.365 | 0.59 | 125.1 | 0.0205 | 0.0213 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1009 | 11/16 | 1.292 | 1.197 | 1.392 | 0.366 | 0.59 | 129.2 | 0.0212 | 0.0220 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1010 | 15/20 | 1.266 | 1.175 | 1.404 | 0.353 | 0.59 | 126.6 | 0.0207 | 0.0216 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1011 | 11/16 | 1.356 | 1.268 | 1.649 | 0.389 | 0.59 | 135.6 | 0.0222 | 0.0231 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1012 | 9/14 | 1.253 | 1.161 | 1.335 | 0.351 | 0.59 | 125.3 | 0.0205 | 0.0214 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1013 | 15/20 | 1.263 | 1.174 | 1.352 | 0.359 | 0.59 | 126.3 | 0.0207 | 0.0215 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1014 | 15/20 | 1.264 | 1.167 | 1.334 | 0.370 | 0.59 | 126.4 | 0.0207 | 0.0216 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1015 | 17/22 | 1.352 | 1.257 | 1.687 | 0.343 | 0.59 | 135.2 | 0.0222 | 0.0230 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1016 | 19/24 | 1.275 | 1.177 | 1.369 | 0.352 | 0.59 | 127.5 | 0.0209 | 0.0217 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1017 | 13/18 | 1.302 | 1.207 | 1.437 | 0.363 | 0.59 | 130.2 | 0.0213 | 0.0222 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1018 | 11/16 | 1.376 | 1.278 | 1.526 | 0.384 | 0.59 | 137.6 | 0.0225 | 0.0234 | Simulator |
| NVIDIA A40 | checkpoint_000300_seed_1019 | 13/18 | 1.349 | 1.252 | 1.475 | 0.367 | 0.59 | 134.9 | 0.0221 | 0.0229 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1000 | 15/20 | 1.298 | 1.216 | 1.547 | 0.330 | 0.59 | 129.8 | 0.0213 | 0.0226 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1001 | 21/26 | 1.346 | 1.259 | 1.474 | 0.362 | 0.59 | 134.6 | 0.0221 | 0.0234 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1002 | 10/15 | 1.318 | 1.226 | 1.442 | 0.370 | 0.59 | 131.8 | 0.0216 | 0.0229 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1003 | 17/22 | 1.357 | 1.270 | 1.735 | 0.350 | 0.59 | 135.7 | 0.0222 | 0.0236 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1004 | 16/21 | 1.248 | 1.160 | 1.245 | 0.358 | 0.59 | 124.8 | 0.0204 | 0.0218 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1005 | 11/16 | 1.270 | 1.183 | 1.414 | 0.339 | 0.59 | 127.0 | 0.0208 | 0.0221 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1006 | 11/16 | 1.274 | 1.190 | 1.342 | 0.330 | 0.59 | 127.4 | 0.0209 | 0.0222 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1007 | 15/20 | 1.302 | 1.218 | 1.385 | 0.369 | 0.59 | 130.2 | 0.0213 | 0.0227 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1008 | 17/22 | 1.218 | 1.130 | 1.215 | 0.342 | 0.59 | 121.8 | 0.0200 | 0.0213 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1009 | 11/16 | 1.232 | 1.144 | 1.288 | 0.357 | 0.59 | 123.2 | 0.0202 | 0.0215 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1010 | 15/20 | 1.214 | 1.126 | 1.227 | 0.352 | 0.59 | 121.4 | 0.0199 | 0.0212 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1011 | 11/16 | 1.211 | 1.126 | 1.225 | 0.337 | 0.59 | 121.1 | 0.0198 | 0.0212 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1012 | 9/14 | 1.190 | 1.101 | 1.153 | 0.345 | 0.59 | 119.0 | 0.0195 | 0.0208 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1013 | 14/19 | 1.198 | 1.113 | 1.141 | 0.360 | 0.59 | 119.8 | 0.0196 | 0.0210 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1014 | 14/19 | 1.200 | 1.113 | 1.140 | 0.354 | 0.59 | 120.0 | 0.0197 | 0.0210 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1015 | 15/20 | 1.220 | 1.135 | 1.184 | 0.364 | 0.59 | 122.0 | 0.0200 | 0.0213 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1016 | 19/24 | 1.206 | 1.118 | 1.158 | 0.364 | 0.59 | 120.6 | 0.0198 | 0.0211 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1017 | 13/18 | 1.205 | 1.117 | 1.163 | 0.359 | 0.59 | 120.5 | 0.0197 | 0.0211 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1018 | 11/16 | 1.202 | 1.115 | 1.139 | 0.357 | 0.59 | 120.2 | 0.0197 | 0.0210 | Simulator |
| NVIDIA A40 | checkpoint_000500_seed_1019 | 14/19 | 1.217 | 1.130 | 1.262 | 0.348 | 0.59 | 121.7 | 0.0199 | 0.0213 | Simulator |
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
| checkpoint_000050_seed_1000_blue | 1.231 | 0.006 | 0.010 | 0.150 | 0.342 | 0.508 | 0.000 | 0.723 |
| checkpoint_000050_seed_1000_red | 1.321 | 0.008 | 0.010 | 0.230 | 0.363 | 0.611 | 0.000 | 0.710 |
| checkpoint_000050_seed_1001_blue | 1.161 | 0.009 | 0.010 | 0.105 | 0.338 | 0.462 | 0.000 | 0.699 |
| checkpoint_000050_seed_1001_red | 1.198 | 0.010 | 0.010 | 0.132 | 0.320 | 0.471 | 0.000 | 0.727 |
| checkpoint_000050_seed_1002_blue | 1.257 | 0.014 | 0.010 | 0.164 | 0.342 | 0.529 | 0.000 | 0.728 |
| checkpoint_000050_seed_1002_red | 1.206 | 0.010 | 0.008 | 0.133 | 0.333 | 0.484 | 0.000 | 0.721 |
| checkpoint_000050_seed_1003_blue | 1.337 | 0.006 | 0.009 | 0.239 | 0.334 | 0.588 | 0.000 | 0.749 |
| checkpoint_000050_seed_1003_red | 1.201 | 0.007 | 0.008 | 0.132 | 0.324 | 0.471 | 0.000 | 0.730 |
| checkpoint_000050_seed_1004_blue | 1.346 | 0.006 | 0.009 | 0.287 | 0.338 | 0.640 | 0.000 | 0.705 |
| checkpoint_000050_seed_1004_red | 1.261 | 0.008 | 0.008 | 0.192 | 0.333 | 0.541 | 0.000 | 0.721 |
| checkpoint_000050_seed_1005_blue | 1.226 | 0.007 | 0.009 | 0.149 | 0.353 | 0.519 | 0.000 | 0.707 |
| checkpoint_000050_seed_1005_red | 1.414 | 0.036 | 0.009 | 0.207 | 0.346 | 0.599 | 0.000 | 0.816 |
| checkpoint_000050_seed_1006_blue | 1.242 | 0.006 | 0.010 | 0.163 | 0.329 | 0.508 | 0.000 | 0.734 |
| checkpoint_000050_seed_1006_red | 1.251 | 0.006 | 0.009 | 0.167 | 0.343 | 0.525 | 0.000 | 0.726 |
| checkpoint_000050_seed_1007_blue | 1.497 | 0.074 | 0.008 | 0.140 | 0.325 | 0.547 | 0.000 | 0.950 |
| checkpoint_000050_seed_1007_red | 1.475 | 0.061 | 0.008 | 0.268 | 0.343 | 0.680 | 0.000 | 0.795 |
| checkpoint_000050_seed_1008_blue | 1.439 | 0.007 | 0.008 | 0.360 | 0.342 | 0.717 | 0.000 | 0.722 |
| checkpoint_000050_seed_1008_red | 1.229 | 0.005 | 0.008 | 0.175 | 0.348 | 0.536 | 0.000 | 0.693 |
| checkpoint_000050_seed_1009_blue | 1.304 | 0.006 | 0.010 | 0.200 | 0.345 | 0.562 | 0.000 | 0.742 |
| checkpoint_000050_seed_1009_red | 1.317 | 0.006 | 0.009 | 0.218 | 0.332 | 0.565 | 0.000 | 0.752 |
| checkpoint_000100_seed_1000_blue | 1.339 | 0.007 | 0.009 | 0.276 | 0.328 | 0.620 | 0.000 | 0.718 |
| checkpoint_000100_seed_1000_red | 1.343 | 0.006 | 0.009 | 0.302 | 0.325 | 0.641 | 0.000 | 0.702 |
| checkpoint_000100_seed_1001_blue | 1.363 | 0.006 | 0.007 | 0.294 | 0.323 | 0.629 | 0.000 | 0.734 |
| checkpoint_000100_seed_1001_red | 1.596 | 0.007 | 0.009 | 0.521 | 0.344 | 0.881 | 0.000 | 0.715 |
| checkpoint_000100_seed_1002_blue | 1.242 | 0.007 | 0.007 | 0.157 | 0.342 | 0.513 | 0.000 | 0.729 |
| checkpoint_000100_seed_1002_red | 1.217 | 0.006 | 0.009 | 0.168 | 0.332 | 0.516 | 0.000 | 0.702 |
| checkpoint_000100_seed_1003_blue | 1.350 | 0.008 | 0.009 | 0.292 | 0.347 | 0.656 | 0.000 | 0.694 |
| checkpoint_000100_seed_1003_red | 1.323 | 0.006 | 0.007 | 0.259 | 0.335 | 0.607 | 0.000 | 0.715 |
| checkpoint_000100_seed_1004_blue | 1.272 | 0.006 | 0.007 | 0.188 | 0.334 | 0.536 | 0.000 | 0.736 |
| checkpoint_000100_seed_1004_red | 1.287 | 0.006 | 0.007 | 0.219 | 0.357 | 0.589 | 0.000 | 0.698 |
| checkpoint_000100_seed_1005_blue | 1.243 | 0.006 | 0.010 | 0.136 | 0.361 | 0.513 | 0.000 | 0.730 |
| checkpoint_000100_seed_1005_red | 1.283 | 0.006 | 0.008 | 0.211 | 0.346 | 0.571 | 0.000 | 0.711 |
| checkpoint_000100_seed_1006_blue | 1.227 | 0.006 | 0.007 | 0.156 | 0.335 | 0.504 | 0.000 | 0.723 |
| checkpoint_000100_seed_1006_red | 1.336 | 0.006 | 0.008 | 0.240 | 0.352 | 0.607 | 0.000 | 0.730 |
| checkpoint_000100_seed_1007_blue | 1.292 | 0.006 | 0.009 | 0.210 | 0.361 | 0.586 | 0.000 | 0.706 |
| checkpoint_000100_seed_1007_red | 1.261 | 0.005 | 0.007 | 0.210 | 0.320 | 0.543 | 0.000 | 0.718 |
| checkpoint_000100_seed_1008_blue | 1.298 | 0.006 | 0.007 | 0.191 | 0.350 | 0.555 | 0.000 | 0.743 |
| checkpoint_000100_seed_1008_red | 1.292 | 0.005 | 0.008 | 0.203 | 0.358 | 0.575 | 0.000 | 0.717 |
| checkpoint_000100_seed_1009_blue | 1.261 | 0.006 | 0.007 | 0.175 | 0.339 | 0.527 | 0.000 | 0.734 |
| checkpoint_000100_seed_1009_red | 1.263 | 0.005 | 0.007 | 0.166 | 0.352 | 0.531 | 0.000 | 0.733 |
| checkpoint_000300_seed_1000_blue | 1.261 | 0.007 | 0.008 | 0.159 | 0.355 | 0.529 | 0.000 | 0.732 |
| checkpoint_000300_seed_1000_red | 1.426 | 0.007 | 0.008 | 0.310 | 0.377 | 0.702 | 0.000 | 0.723 |
| checkpoint_000300_seed_1001_blue | 1.228 | 0.006 | 0.007 | 0.160 | 0.353 | 0.527 | 0.000 | 0.701 |
| checkpoint_000300_seed_1001_red | 1.239 | 0.006 | 0.008 | 0.166 | 0.332 | 0.513 | 0.000 | 0.726 |
| checkpoint_000300_seed_1002_blue | 1.254 | 0.006 | 0.008 | 0.181 | 0.347 | 0.542 | 0.000 | 0.713 |
| checkpoint_000300_seed_1002_red | 1.274 | 0.006 | 0.009 | 0.204 | 0.349 | 0.568 | 0.000 | 0.707 |
| checkpoint_000300_seed_1003_blue | 1.218 | 0.006 | 0.007 | 0.137 | 0.358 | 0.508 | 0.000 | 0.711 |
| checkpoint_000300_seed_1003_red | 1.273 | 0.006 | 0.007 | 0.211 | 0.349 | 0.573 | 0.000 | 0.700 |
| checkpoint_000300_seed_1004_blue | 1.291 | 0.006 | 0.008 | 0.207 | 0.354 | 0.575 | 0.000 | 0.716 |
| checkpoint_000300_seed_1004_red | 1.267 | 0.006 | 0.008 | 0.181 | 0.357 | 0.552 | 0.000 | 0.716 |
| checkpoint_000300_seed_1005_blue | 1.520 | 0.008 | 0.008 | 0.397 | 0.356 | 0.769 | 0.000 | 0.751 |
| checkpoint_000300_seed_1005_red | 1.236 | 0.007 | 0.006 | 0.146 | 0.325 | 0.484 | 0.000 | 0.751 |
| checkpoint_000300_seed_1006_blue | 1.241 | 0.007 | 0.008 | 0.194 | 0.328 | 0.538 | 0.000 | 0.703 |
| checkpoint_000300_seed_1006_red | 1.318 | 0.008 | 0.007 | 0.223 | 0.360 | 0.598 | 0.000 | 0.720 |
| checkpoint_000300_seed_1007_blue | 1.258 | 0.006 | 0.007 | 0.187 | 0.338 | 0.539 | 0.000 | 0.720 |
| checkpoint_000300_seed_1007_red | 1.278 | 0.007 | 0.008 | 0.203 | 0.349 | 0.568 | 0.000 | 0.710 |
| checkpoint_000300_seed_1008_blue | 1.239 | 0.007 | 0.008 | 0.163 | 0.342 | 0.520 | 0.000 | 0.718 |
| checkpoint_000300_seed_1008_red | 1.284 | 0.006 | 0.008 | 0.173 | 0.365 | 0.552 | 0.000 | 0.731 |
| checkpoint_000300_seed_1009_blue | 1.256 | 0.006 | 0.007 | 0.179 | 0.361 | 0.553 | 0.000 | 0.703 |
| checkpoint_000300_seed_1009_red | 1.218 | 0.007 | 0.007 | 0.152 | 0.337 | 0.503 | 0.000 | 0.715 |
| checkpoint_000500_seed_1000_blue | 1.318 | 0.006 | 0.009 | 0.258 | 0.334 | 0.606 | 0.000 | 0.712 |
| checkpoint_000500_seed_1000_red | 1.343 | 0.005 | 0.010 | 0.265 | 0.345 | 0.626 | 0.000 | 0.717 |
| checkpoint_000500_seed_1001_blue | 1.273 | 0.007 | 0.009 | 0.191 | 0.345 | 0.551 | 0.000 | 0.722 |
| checkpoint_000500_seed_1001_red | 1.251 | 0.007 | 0.010 | 0.171 | 0.329 | 0.518 | 0.000 | 0.733 |
| checkpoint_000500_seed_1002_blue | 1.203 | 0.007 | 0.010 | 0.150 | 0.325 | 0.491 | 0.000 | 0.712 |
| checkpoint_000500_seed_1002_red | 1.198 | 0.006 | 0.011 | 0.117 | 0.339 | 0.473 | 0.000 | 0.725 |
| checkpoint_000500_seed_1003_blue | 1.309 | 0.006 | 0.010 | 0.237 | 0.339 | 0.593 | 0.000 | 0.716 |
| checkpoint_000500_seed_1003_red | 1.234 | 0.007 | 0.009 | 0.166 | 0.327 | 0.509 | 0.000 | 0.725 |
| checkpoint_000500_seed_1004_blue | 1.195 | 0.006 | 0.009 | 0.128 | 0.329 | 0.472 | 0.000 | 0.723 |
| checkpoint_000500_seed_1004_red | 1.302 | 0.058 | 0.009 | 0.109 | 0.332 | 0.507 | 0.000 | 0.795 |
| checkpoint_000500_seed_1005_blue | 1.227 | 0.007 | 0.010 | 0.127 | 0.352 | 0.495 | 0.000 | 0.732 |
| checkpoint_000500_seed_1005_red | 1.261 | 0.006 | 0.010 | 0.199 | 0.320 | 0.535 | 0.000 | 0.726 |
| checkpoint_000500_seed_1006_blue | 1.194 | 0.006 | 0.010 | 0.091 | 0.347 | 0.455 | 0.000 | 0.739 |
| checkpoint_000500_seed_1006_red | 1.263 | 0.007 | 0.010 | 0.172 | 0.329 | 0.519 | 0.000 | 0.744 |
| checkpoint_000500_seed_1007_blue | 1.197 | 0.006 | 0.010 | 0.124 | 0.335 | 0.475 | 0.000 | 0.722 |
| checkpoint_000500_seed_1007_red | 1.213 | 0.006 | 0.009 | 0.119 | 0.336 | 0.470 | 0.000 | 0.743 |
| checkpoint_000500_seed_1008_blue | 1.214 | 0.006 | 0.009 | 0.132 | 0.345 | 0.492 | 0.000 | 0.722 |
| checkpoint_000500_seed_1008_red | 1.248 | 0.006 | 0.009 | 0.146 | 0.347 | 0.509 | 0.000 | 0.739 |
| checkpoint_000500_seed_1009_blue | 1.238 | 0.006 | 0.009 | 0.166 | 0.346 | 0.526 | 0.000 | 0.712 |
| checkpoint_000500_seed_1009_red | 1.209 | 0.005 | 0.009 | 0.123 | 0.362 | 0.499 | 0.000 | 0.710 |
| checkpoint_000050_seed_1000_black-purple_black | 1.208 | 0.006 | 0.011 | 0.118 | 0.366 | 0.502 | 0.000 | 0.706 |
| checkpoint_000050_seed_1000_black-purple_purple | 1.161 | 0.006 | 0.008 | 0.083 | 0.352 | 0.450 | 0.000 | 0.711 |
| checkpoint_000050_seed_1000_green-yellow_green | 1.143 | 0.007 | 0.009 | 0.061 | 0.329 | 0.406 | 0.000 | 0.737 |
| checkpoint_000050_seed_1000_green-yellow_yellow | 1.180 | 0.007 | 0.009 | 0.076 | 0.340 | 0.432 | 0.000 | 0.748 |
| checkpoint_000050_seed_1001_black-purple_black | 1.160 | 0.006 | 0.009 | 0.084 | 0.350 | 0.448 | 0.000 | 0.712 |
| checkpoint_000050_seed_1001_black-purple_purple | 1.242 | 0.006 | 0.010 | 0.135 | 0.367 | 0.518 | 0.000 | 0.724 |
| checkpoint_000050_seed_1001_green-yellow_green | 1.210 | 0.006 | 0.010 | 0.099 | 0.349 | 0.465 | 0.000 | 0.745 |
| checkpoint_000050_seed_1001_green-yellow_yellow | 1.197 | 0.007 | 0.010 | 0.085 | 0.352 | 0.454 | 0.000 | 0.743 |
| checkpoint_000050_seed_1002_black-purple_black | 1.251 | 0.005 | 0.010 | 0.162 | 0.351 | 0.528 | 0.000 | 0.724 |
| checkpoint_000050_seed_1002_black-purple_purple | 1.158 | 0.005 | 0.009 | 0.071 | 0.344 | 0.429 | 0.000 | 0.728 |
| checkpoint_000050_seed_1002_green-yellow_green | 1.223 | 0.006 | 0.009 | 0.078 | 0.338 | 0.431 | 0.000 | 0.792 |
| checkpoint_000050_seed_1002_green-yellow_yellow | 1.193 | 0.007 | 0.009 | 0.069 | 0.333 | 0.417 | 0.000 | 0.776 |
| checkpoint_000050_seed_1003_black-purple_black | 1.150 | 0.006 | 0.009 | 0.087 | 0.351 | 0.453 | 0.000 | 0.696 |
| checkpoint_000050_seed_1003_black-purple_purple | 1.178 | 0.007 | 0.010 | 0.068 | 0.360 | 0.444 | 0.000 | 0.734 |
| checkpoint_000050_seed_1003_green-yellow_green | 1.218 | 0.006 | 0.010 | 0.087 | 0.343 | 0.446 | 0.000 | 0.772 |
| checkpoint_000050_seed_1003_green-yellow_yellow | 1.152 | 0.006 | 0.008 | 0.079 | 0.326 | 0.418 | 0.000 | 0.733 |
| checkpoint_000050_seed_1004_black-purple_black | 1.183 | 0.006 | 0.011 | 0.068 | 0.366 | 0.450 | 0.000 | 0.732 |
| checkpoint_000050_seed_1004_black-purple_purple | 1.148 | 0.006 | 0.010 | 0.067 | 0.352 | 0.435 | 0.000 | 0.713 |
| checkpoint_000050_seed_1004_green-yellow_green | 1.162 | 0.005 | 0.009 | 0.091 | 0.334 | 0.440 | 0.000 | 0.723 |
| checkpoint_000050_seed_1004_green-yellow_yellow | 1.178 | 0.006 | 0.010 | 0.111 | 0.329 | 0.456 | 0.000 | 0.722 |
| checkpoint_000050_seed_1005_black-purple_black | 1.136 | 0.006 | 0.010 | 0.072 | 0.346 | 0.434 | 0.000 | 0.702 |
| checkpoint_000050_seed_1005_black-purple_purple | 1.156 | 0.006 | 0.010 | 0.077 | 0.360 | 0.452 | 0.000 | 0.705 |
| checkpoint_000050_seed_1006_black-purple_black | 1.487 | 0.243 | 0.008 | 0.074 | 0.324 | 0.648 | 0.000 | 0.839 |
| checkpoint_000050_seed_1006_black-purple_purple | 1.133 | 0.006 | 0.009 | 0.083 | 0.326 | 0.424 | 0.000 | 0.709 |
| checkpoint_000050_seed_1007_black-purple_black | 1.177 | 0.005 | 0.009 | 0.071 | 0.335 | 0.421 | 0.000 | 0.756 |
| checkpoint_000050_seed_1007_black-purple_purple | 1.163 | 0.007 | 0.008 | 0.084 | 0.324 | 0.423 | 0.000 | 0.740 |
| checkpoint_000050_seed_1008_black-purple_black | 1.176 | 0.006 | 0.009 | 0.101 | 0.360 | 0.475 | 0.000 | 0.701 |
| checkpoint_000050_seed_1008_black-purple_purple | 1.154 | 0.007 | 0.009 | 0.094 | 0.338 | 0.448 | 0.000 | 0.707 |
| checkpoint_000050_seed_1009_black-purple_black | 1.184 | 0.007 | 0.010 | 0.094 | 0.344 | 0.455 | 0.000 | 0.729 |
| checkpoint_000050_seed_1009_black-purple_purple | 1.200 | 0.007 | 0.009 | 0.110 | 0.344 | 0.470 | 0.000 | 0.730 |
| checkpoint_000050_seed_1000 | 1.109 | 0.005 | 0.010 | 0.036 | 0.318 | 0.369 | 0.000 | 0.740 |
| checkpoint_000050_seed_1001 | 1.190 | 0.006 | 0.011 | 0.113 | 0.325 | 0.455 | 0.000 | 0.735 |
| checkpoint_000050_seed_1002 | 1.180 | 0.006 | 0.010 | 0.123 | 0.321 | 0.459 | 0.000 | 0.720 |
| checkpoint_000050_seed_1003 | 1.190 | 0.006 | 0.010 | 0.110 | 0.334 | 0.460 | 0.000 | 0.730 |
| checkpoint_000050_seed_1004 | 1.155 | 0.006 | 0.010 | 0.064 | 0.345 | 0.425 | 0.000 | 0.731 |
| checkpoint_000050_seed_1005 | 1.184 | 0.006 | 0.009 | 0.140 | 0.316 | 0.471 | 0.000 | 0.712 |
| checkpoint_000050_seed_1006 | 1.185 | 0.006 | 0.010 | 0.073 | 0.353 | 0.441 | 0.000 | 0.744 |
| checkpoint_000050_seed_1007 | 1.180 | 0.032 | 0.010 | 0.069 | 0.319 | 0.429 | 0.000 | 0.751 |
| checkpoint_000050_seed_1008 | 1.120 | 0.006 | 0.007 | 0.053 | 0.326 | 0.393 | 0.000 | 0.728 |
| checkpoint_000050_seed_1009 | 1.148 | 0.006 | 0.010 | 0.091 | 0.326 | 0.433 | 0.000 | 0.715 |
| checkpoint_000050_seed_1010 | 1.149 | 0.005 | 0.008 | 0.070 | 0.322 | 0.406 | 0.000 | 0.744 |
| checkpoint_000050_seed_1011 | 1.158 | 0.006 | 0.008 | 0.097 | 0.319 | 0.430 | 0.000 | 0.728 |
| checkpoint_000050_seed_1012 | 1.151 | 0.006 | 0.008 | 0.052 | 0.326 | 0.392 | 0.000 | 0.759 |
| checkpoint_000050_seed_1013 | 1.093 | 0.006 | 0.008 | 0.039 | 0.304 | 0.357 | 0.000 | 0.735 |
| checkpoint_000050_seed_1014 | 1.135 | 0.006 | 0.010 | 0.047 | 0.334 | 0.398 | 0.000 | 0.737 |
| checkpoint_000050_seed_1015 | 1.105 | 0.006 | 0.008 | 0.041 | 0.334 | 0.390 | 0.000 | 0.716 |
| checkpoint_000050_seed_1016 | 1.113 | 0.006 | 0.009 | 0.063 | 0.328 | 0.405 | 0.000 | 0.708 |
| checkpoint_000050_seed_1017 | 1.134 | 0.006 | 0.010 | 0.061 | 0.346 | 0.422 | 0.000 | 0.712 |
| checkpoint_000050_seed_1018 | 1.101 | 0.005 | 0.010 | 0.048 | 0.328 | 0.391 | 0.000 | 0.710 |
| checkpoint_000050_seed_1019 | 1.121 | 0.006 | 0.009 | 0.063 | 0.331 | 0.409 | 0.000 | 0.711 |
| checkpoint_000100_seed_1000 | 1.516 | 0.006 | 0.011 | 0.435 | 0.342 | 0.793 | 0.000 | 0.723 |
| checkpoint_000100_seed_1001 | 1.208 | 0.006 | 0.010 | 0.116 | 0.342 | 0.473 | 0.000 | 0.734 |
| checkpoint_000100_seed_1002 | 1.218 | 0.006 | 0.011 | 0.133 | 0.346 | 0.495 | 0.000 | 0.722 |
| checkpoint_000100_seed_1003 | 1.218 | 0.006 | 0.009 | 0.131 | 0.340 | 0.487 | 0.000 | 0.731 |
| checkpoint_000100_seed_1004 | 1.228 | 0.005 | 0.011 | 0.126 | 0.361 | 0.503 | 0.000 | 0.725 |
| checkpoint_000100_seed_1005 | 1.253 | 0.005 | 0.010 | 0.167 | 0.344 | 0.526 | 0.000 | 0.727 |
| checkpoint_000100_seed_1006 | 1.193 | 0.005 | 0.009 | 0.115 | 0.331 | 0.460 | 0.000 | 0.733 |
| checkpoint_000100_seed_1007 | 1.164 | 0.006 | 0.010 | 0.069 | 0.341 | 0.425 | 0.000 | 0.738 |
| checkpoint_000100_seed_1008 | 1.138 | 0.006 | 0.011 | 0.038 | 0.356 | 0.411 | 0.000 | 0.727 |
| checkpoint_000100_seed_1009 | 1.145 | 0.006 | 0.009 | 0.096 | 0.324 | 0.435 | 0.000 | 0.710 |
| checkpoint_000100_seed_1010 | 1.214 | 0.006 | 0.010 | 0.120 | 0.348 | 0.484 | 0.000 | 0.729 |
| checkpoint_000100_seed_1011 | 1.125 | 0.005 | 0.008 | 0.053 | 0.329 | 0.396 | 0.000 | 0.729 |
| checkpoint_000100_seed_1012 | 1.170 | 0.005 | 0.009 | 0.073 | 0.341 | 0.428 | 0.000 | 0.743 |
| checkpoint_000100_seed_1013 | 1.219 | 0.005 | 0.009 | 0.130 | 0.349 | 0.494 | 0.000 | 0.726 |
| checkpoint_000100_seed_1014 | 1.208 | 0.005 | 0.009 | 0.121 | 0.329 | 0.465 | 0.000 | 0.744 |
| checkpoint_000100_seed_1015 | 1.244 | 0.006 | 0.010 | 0.140 | 0.353 | 0.508 | 0.000 | 0.736 |
| checkpoint_000100_seed_1016 | 1.131 | 0.006 | 0.009 | 0.062 | 0.326 | 0.403 | 0.000 | 0.729 |
| checkpoint_000100_seed_1017 | 2.048 | 0.613 | 0.010 | 0.114 | 0.331 | 1.069 | 0.000 | 0.980 |
| checkpoint_000100_seed_1018 | 1.226 | 0.005 | 0.010 | 0.122 | 0.347 | 0.484 | 0.000 | 0.742 |
| checkpoint_000100_seed_1019 | 1.390 | 0.151 | 0.010 | 0.043 | 0.348 | 0.551 | 0.000 | 0.838 |
| checkpoint_000300_seed_1000 | 1.188 | 0.013 | 0.011 | 0.040 | 0.359 | 0.424 | 0.000 | 0.764 |
| checkpoint_000300_seed_1001 | 1.144 | 0.010 | 0.011 | 0.033 | 0.347 | 0.401 | 0.000 | 0.743 |
| checkpoint_000300_seed_1002 | 1.104 | 0.009 | 0.011 | 0.023 | 0.331 | 0.374 | 0.000 | 0.730 |
| checkpoint_000300_seed_1003 | 1.181 | 0.012 | 0.010 | 0.087 | 0.333 | 0.442 | 0.000 | 0.739 |
| checkpoint_000300_seed_1004 | 1.182 | 0.008 | 0.011 | 0.083 | 0.342 | 0.445 | 0.000 | 0.737 |
| checkpoint_000300_seed_1005 | 1.198 | 0.010 | 0.010 | 0.084 | 0.363 | 0.468 | 0.000 | 0.730 |
| checkpoint_000300_seed_1006 | 1.193 | 0.011 | 0.010 | 0.111 | 0.346 | 0.479 | 0.000 | 0.715 |
| checkpoint_000300_seed_1007 | 1.176 | 0.011 | 0.011 | 0.079 | 0.355 | 0.457 | 0.000 | 0.719 |
| checkpoint_000300_seed_1008 | 1.157 | 0.013 | 0.010 | 0.051 | 0.365 | 0.439 | 0.000 | 0.718 |
| checkpoint_000300_seed_1009 | 1.197 | 0.012 | 0.010 | 0.088 | 0.366 | 0.476 | 0.000 | 0.721 |
| checkpoint_000300_seed_1010 | 1.175 | 0.009 | 0.011 | 0.086 | 0.353 | 0.459 | 0.000 | 0.716 |
| checkpoint_000300_seed_1011 | 1.268 | 0.008 | 0.011 | 0.142 | 0.389 | 0.550 | 0.000 | 0.718 |
| checkpoint_000300_seed_1012 | 1.161 | 0.008 | 0.012 | 0.059 | 0.351 | 0.430 | 0.000 | 0.731 |
| checkpoint_000300_seed_1013 | 1.174 | 0.010 | 0.011 | 0.064 | 0.359 | 0.444 | 0.000 | 0.730 |
| checkpoint_000300_seed_1014 | 1.167 | 0.011 | 0.011 | 0.056 | 0.370 | 0.448 | 0.000 | 0.718 |
| checkpoint_000300_seed_1015 | 1.257 | 0.082 | 0.010 | 0.041 | 0.343 | 0.476 | 0.000 | 0.782 |
| checkpoint_000300_seed_1016 | 1.177 | 0.011 | 0.010 | 0.069 | 0.352 | 0.442 | 0.000 | 0.735 |
| checkpoint_000300_seed_1017 | 1.207 | 0.011 | 0.011 | 0.071 | 0.363 | 0.455 | 0.000 | 0.752 |
| checkpoint_000300_seed_1018 | 1.278 | 0.012 | 0.012 | 0.125 | 0.384 | 0.533 | 0.000 | 0.745 |
| checkpoint_000300_seed_1019 | 1.252 | 0.012 | 0.012 | 0.087 | 0.367 | 0.478 | 0.000 | 0.774 |
| checkpoint_000500_seed_1000 | 1.216 | 0.006 | 0.010 | 0.125 | 0.330 | 0.471 | 0.000 | 0.745 |
| checkpoint_000500_seed_1001 | 1.259 | 0.006 | 0.011 | 0.170 | 0.362 | 0.549 | 0.000 | 0.710 |
| checkpoint_000500_seed_1002 | 1.226 | 0.006 | 0.012 | 0.120 | 0.370 | 0.508 | 0.000 | 0.718 |
| checkpoint_000500_seed_1003 | 1.270 | 0.062 | 0.009 | 0.074 | 0.350 | 0.495 | 0.000 | 0.775 |
| checkpoint_000500_seed_1004 | 1.160 | 0.006 | 0.011 | 0.038 | 0.358 | 0.412 | 0.000 | 0.748 |
| checkpoint_000500_seed_1005 | 1.183 | 0.006 | 0.010 | 0.108 | 0.339 | 0.464 | 0.000 | 0.719 |
| checkpoint_000500_seed_1006 | 1.190 | 0.006 | 0.010 | 0.118 | 0.330 | 0.465 | 0.000 | 0.726 |
| checkpoint_000500_seed_1007 | 1.218 | 0.006 | 0.011 | 0.100 | 0.369 | 0.486 | 0.000 | 0.732 |
| checkpoint_000500_seed_1008 | 1.130 | 0.006 | 0.011 | 0.030 | 0.342 | 0.388 | 0.000 | 0.742 |
| checkpoint_000500_seed_1009 | 1.144 | 0.006 | 0.010 | 0.047 | 0.357 | 0.420 | 0.000 | 0.724 |
| checkpoint_000500_seed_1010 | 1.126 | 0.005 | 0.011 | 0.037 | 0.352 | 0.405 | 0.000 | 0.721 |
| checkpoint_000500_seed_1011 | 1.126 | 0.006 | 0.010 | 0.048 | 0.337 | 0.401 | 0.000 | 0.725 |
| checkpoint_000500_seed_1012 | 1.101 | 0.005 | 0.009 | 0.031 | 0.345 | 0.390 | 0.000 | 0.711 |
| checkpoint_000500_seed_1013 | 1.113 | 0.006 | 0.011 | 0.026 | 0.360 | 0.403 | 0.000 | 0.710 |
| checkpoint_000500_seed_1014 | 1.113 | 0.006 | 0.011 | 0.032 | 0.354 | 0.404 | 0.000 | 0.710 |
| checkpoint_000500_seed_1015 | 1.135 | 0.006 | 0.012 | 0.031 | 0.364 | 0.412 | 0.000 | 0.723 |
| checkpoint_000500_seed_1016 | 1.118 | 0.005 | 0.011 | 0.025 | 0.364 | 0.405 | 0.000 | 0.713 |
| checkpoint_000500_seed_1017 | 1.117 | 0.006 | 0.011 | 0.029 | 0.359 | 0.405 | 0.000 | 0.712 |
| checkpoint_000500_seed_1018 | 1.115 | 0.006 | 0.011 | 0.033 | 0.357 | 0.407 | 0.000 | 0.708 |
| checkpoint_000500_seed_1019 | 1.130 | 0.006 | 0.011 | 0.050 | 0.348 | 0.414 | 0.000 | 0.716 |
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
| checkpoint_000050_seed_1000_blue | 0.000 | 0.001 | 0.001 | 1.230 | 0.000 |
| checkpoint_000050_seed_1000_red | 0.000 | 0.001 | 0.001 | 1.320 | 0.000 |
| checkpoint_000050_seed_1001_blue | 0.001 | 0.001 | 0.001 | 1.160 | 0.000 |
| checkpoint_000050_seed_1001_red | 0.000 | 0.001 | 0.000 | 1.198 | 0.000 |
| checkpoint_000050_seed_1002_blue | 0.000 | 0.001 | 0.001 | 1.257 | 0.000 |
| checkpoint_000050_seed_1002_red | 0.001 | 0.001 | 0.000 | 1.205 | 0.000 |
| checkpoint_000050_seed_1003_blue | 0.001 | 0.001 | 0.001 | 1.336 | 0.000 |
| checkpoint_000050_seed_1003_red | 0.001 | 0.001 | 0.001 | 1.200 | 0.000 |
| checkpoint_000050_seed_1004_blue | 0.000 | 0.001 | 0.000 | 1.345 | 0.000 |
| checkpoint_000050_seed_1004_red | 0.000 | 0.001 | 0.000 | 1.261 | 0.000 |
| checkpoint_000050_seed_1005_blue | 0.001 | 0.001 | 0.001 | 1.225 | 0.000 |
| checkpoint_000050_seed_1005_red | 0.000 | 0.001 | 0.000 | 1.414 | 0.000 |
| checkpoint_000050_seed_1006_blue | 0.000 | 0.001 | 0.000 | 1.241 | 0.000 |
| checkpoint_000050_seed_1006_red | 0.000 | 0.001 | 0.001 | 1.250 | 0.000 |
| checkpoint_000050_seed_1007_blue | 0.000 | 0.001 | 0.000 | 1.497 | 0.000 |
| checkpoint_000050_seed_1007_red | 0.001 | 0.001 | 0.001 | 1.474 | 0.000 |
| checkpoint_000050_seed_1008_blue | 0.000 | 0.001 | 0.001 | 1.438 | 0.000 |
| checkpoint_000050_seed_1008_red | 0.001 | 0.001 | 0.000 | 1.228 | 0.000 |
| checkpoint_000050_seed_1009_blue | 0.000 | 0.001 | 0.001 | 1.303 | 0.000 |
| checkpoint_000050_seed_1009_red | 0.001 | 0.001 | 0.001 | 1.316 | 0.000 |
| checkpoint_000100_seed_1000_blue | 0.000 | 0.001 | 0.000 | 1.338 | 0.000 |
| checkpoint_000100_seed_1000_red | 0.001 | 0.001 | 0.001 | 1.343 | 0.000 |
| checkpoint_000100_seed_1001_blue | 0.000 | 0.001 | 0.000 | 1.362 | 0.000 |
| checkpoint_000100_seed_1001_red | 0.001 | 0.001 | 0.001 | 1.595 | 0.000 |
| checkpoint_000100_seed_1002_blue | 0.000 | 0.001 | 0.001 | 1.241 | 0.000 |
| checkpoint_000100_seed_1002_red | 0.000 | 0.001 | 0.000 | 1.217 | 0.000 |
| checkpoint_000100_seed_1003_blue | 0.000 | 0.001 | 0.001 | 1.349 | 0.000 |
| checkpoint_000100_seed_1003_red | 0.000 | 0.001 | 0.001 | 1.322 | 0.000 |
| checkpoint_000100_seed_1004_blue | 0.001 | 0.001 | 0.001 | 1.258 | 0.013 |
| checkpoint_000100_seed_1004_red | 0.000 | 0.001 | 0.001 | 1.286 | 0.000 |
| checkpoint_000100_seed_1005_blue | 0.000 | 0.001 | 0.001 | 1.242 | 0.000 |
| checkpoint_000100_seed_1005_red | 0.001 | 0.001 | 0.001 | 1.282 | 0.000 |
| checkpoint_000100_seed_1006_blue | 0.000 | 0.001 | 0.001 | 1.226 | 0.000 |
| checkpoint_000100_seed_1006_red | 0.000 | 0.001 | 0.001 | 1.316 | 0.019 |
| checkpoint_000100_seed_1007_blue | 0.000 | 0.001 | 0.001 | 1.291 | 0.000 |
| checkpoint_000100_seed_1007_red | 0.000 | 0.001 | 0.001 | 1.260 | 0.000 |
| checkpoint_000100_seed_1008_blue | 0.000 | 0.001 | 0.001 | 1.275 | 0.022 |
| checkpoint_000100_seed_1008_red | 0.000 | 0.001 | 0.001 | 1.291 | 0.000 |
| checkpoint_000100_seed_1009_blue | 0.000 | 0.001 | 0.001 | 1.260 | 0.000 |
| checkpoint_000100_seed_1009_red | 0.000 | 0.001 | 0.001 | 1.262 | 0.000 |
| checkpoint_000300_seed_1000_blue | 0.000 | 0.001 | 0.001 | 1.261 | 0.000 |
| checkpoint_000300_seed_1000_red | 0.000 | 0.001 | 0.001 | 1.425 | 0.000 |
| checkpoint_000300_seed_1001_blue | 0.000 | 0.001 | 0.000 | 1.227 | 0.000 |
| checkpoint_000300_seed_1001_red | 0.000 | 0.001 | 0.000 | 1.238 | 0.000 |
| checkpoint_000300_seed_1002_blue | 0.000 | 0.001 | 0.000 | 1.254 | 0.000 |
| checkpoint_000300_seed_1002_red | 0.000 | 0.001 | 0.000 | 1.274 | 0.000 |
| checkpoint_000300_seed_1003_blue | 0.000 | 0.001 | 0.000 | 1.218 | 0.000 |
| checkpoint_000300_seed_1003_red | 0.000 | 0.001 | 0.000 | 1.273 | 0.000 |
| checkpoint_000300_seed_1004_blue | 0.000 | 0.001 | 0.000 | 1.290 | 0.000 |
| checkpoint_000300_seed_1004_red | 0.000 | 0.001 | 0.000 | 1.267 | 0.000 |
| checkpoint_000300_seed_1005_blue | 0.001 | 0.001 | 0.001 | 1.500 | 0.019 |
| checkpoint_000300_seed_1005_red | 0.000 | 0.001 | 0.001 | 1.213 | 0.022 |
| checkpoint_000300_seed_1006_blue | 0.000 | 0.001 | 0.001 | 1.240 | 0.000 |
| checkpoint_000300_seed_1006_red | 0.000 | 0.001 | 0.001 | 1.317 | 0.000 |
| checkpoint_000300_seed_1007_blue | 0.000 | 0.001 | 0.001 | 1.257 | 0.000 |
| checkpoint_000300_seed_1007_red | 0.000 | 0.001 | 0.001 | 1.276 | 0.000 |
| checkpoint_000300_seed_1008_blue | 0.000 | 0.001 | 0.001 | 1.238 | 0.000 |
| checkpoint_000300_seed_1008_red | 0.000 | 0.001 | 0.001 | 1.264 | 0.019 |
| checkpoint_000300_seed_1009_blue | 0.000 | 0.001 | 0.001 | 1.255 | 0.000 |
| checkpoint_000300_seed_1009_red | 0.000 | 0.001 | 0.001 | 1.217 | 0.000 |
| checkpoint_000500_seed_1000_blue | 0.000 | 0.001 | 0.001 | 1.317 | 0.000 |
| checkpoint_000500_seed_1000_red | 0.000 | 0.001 | 0.001 | 1.342 | 0.000 |
| checkpoint_000500_seed_1001_blue | 0.001 | 0.001 | 0.001 | 1.272 | 0.000 |
| checkpoint_000500_seed_1001_red | 0.000 | 0.001 | 0.001 | 1.250 | 0.000 |
| checkpoint_000500_seed_1002_blue | 0.000 | 0.001 | 0.001 | 1.202 | 0.000 |
| checkpoint_000500_seed_1002_red | 0.000 | 0.001 | 0.001 | 1.197 | 0.000 |
| checkpoint_000500_seed_1003_blue | 0.000 | 0.001 | 0.001 | 1.308 | 0.000 |
| checkpoint_000500_seed_1003_red | 0.000 | 0.001 | 0.001 | 1.233 | 0.000 |
| checkpoint_000500_seed_1004_blue | 0.000 | 0.001 | 0.001 | 1.194 | 0.000 |
| checkpoint_000500_seed_1004_red | 0.000 | 0.001 | 0.001 | 1.301 | 0.000 |
| checkpoint_000500_seed_1005_blue | 0.000 | 0.001 | 0.001 | 1.226 | 0.000 |
| checkpoint_000500_seed_1005_red | 0.000 | 0.001 | 0.001 | 1.259 | 0.000 |
| checkpoint_000500_seed_1006_blue | 0.000 | 0.001 | 0.001 | 1.193 | 0.000 |
| checkpoint_000500_seed_1006_red | 0.000 | 0.001 | 0.001 | 1.261 | 0.000 |
| checkpoint_000500_seed_1007_blue | 0.001 | 0.001 | 0.001 | 1.196 | 0.000 |
| checkpoint_000500_seed_1007_red | 0.000 | 0.001 | 0.001 | 1.212 | 0.000 |
| checkpoint_000500_seed_1008_blue | 0.000 | 0.001 | 0.001 | 1.213 | 0.000 |
| checkpoint_000500_seed_1008_red | 0.000 | 0.001 | 0.001 | 1.247 | 0.000 |
| checkpoint_000500_seed_1009_blue | 0.000 | 0.001 | 0.001 | 1.237 | 0.000 |
| checkpoint_000500_seed_1009_red | 0.000 | 0.001 | 0.001 | 1.208 | 0.000 |
| checkpoint_000050_seed_1000_black-purple_black | 0.000 | 0.003 | 0.001 | 1.207 | 0.000 |
| checkpoint_000050_seed_1000_black-purple_purple | 0.000 | 0.004 | 0.001 | 1.160 | 0.000 |
| checkpoint_000050_seed_1000_green-yellow_green | 0.000 | 0.004 | 0.001 | 1.142 | 0.000 |
| checkpoint_000050_seed_1000_green-yellow_yellow | 0.000 | 0.003 | 0.001 | 1.179 | 0.000 |
| checkpoint_000050_seed_1001_black-purple_black | 0.001 | 0.003 | 0.001 | 1.159 | 0.000 |
| checkpoint_000050_seed_1001_black-purple_purple | 0.001 | 0.003 | 0.001 | 1.241 | 0.000 |
| checkpoint_000050_seed_1001_green-yellow_green | 0.000 | 0.004 | 0.001 | 1.209 | 0.000 |
| checkpoint_000050_seed_1001_green-yellow_yellow | 0.001 | 0.004 | 0.001 | 1.196 | 0.000 |
| checkpoint_000050_seed_1002_black-purple_black | 0.001 | 0.003 | 0.000 | 1.251 | 0.000 |
| checkpoint_000050_seed_1002_black-purple_purple | 0.001 | 0.004 | 0.001 | 1.157 | 0.000 |
| checkpoint_000050_seed_1002_green-yellow_green | 0.000 | 0.004 | 0.001 | 1.223 | 0.000 |
| checkpoint_000050_seed_1002_green-yellow_yellow | 0.000 | 0.003 | 0.001 | 1.192 | 0.000 |
| checkpoint_000050_seed_1003_black-purple_black | 0.000 | 0.004 | 0.001 | 1.149 | 0.000 |
| checkpoint_000050_seed_1003_black-purple_purple | 0.001 | 0.004 | 0.001 | 1.177 | 0.000 |
| checkpoint_000050_seed_1003_green-yellow_green | 0.000 | 0.002 | 0.000 | 1.217 | 0.000 |
| checkpoint_000050_seed_1003_green-yellow_yellow | 0.001 | 0.002 | 0.000 | 1.152 | 0.000 |
| checkpoint_000050_seed_1004_black-purple_black | 0.000 | 0.004 | 0.001 | 1.182 | 0.000 |
| checkpoint_000050_seed_1004_black-purple_purple | 0.000 | 0.004 | 0.001 | 1.147 | 0.000 |
| checkpoint_000050_seed_1004_green-yellow_green | 0.000 | 0.002 | 0.000 | 1.162 | 0.000 |
| checkpoint_000050_seed_1004_green-yellow_yellow | 0.001 | 0.003 | 0.000 | 1.177 | 0.000 |
| checkpoint_000050_seed_1005_black-purple_black | 0.000 | 0.004 | 0.001 | 1.135 | 0.000 |
| checkpoint_000050_seed_1005_black-purple_purple | 0.000 | 0.004 | 0.001 | 1.155 | 0.000 |
| checkpoint_000050_seed_1006_black-purple_black | 0.001 | 0.003 | 0.001 | 1.486 | 0.000 |
| checkpoint_000050_seed_1006_black-purple_purple | 0.000 | 0.003 | 0.001 | 1.132 | 0.000 |
| checkpoint_000050_seed_1007_black-purple_black | 0.000 | 0.004 | 0.001 | 1.176 | 0.000 |
| checkpoint_000050_seed_1007_black-purple_purple | 0.000 | 0.004 | 0.001 | 1.162 | 0.000 |
| checkpoint_000050_seed_1008_black-purple_black | 0.001 | 0.004 | 0.001 | 1.175 | 0.000 |
| checkpoint_000050_seed_1008_black-purple_purple | 0.000 | 0.003 | 0.000 | 1.154 | 0.000 |
| checkpoint_000050_seed_1009_black-purple_black | 0.000 | 0.004 | 0.001 | 1.183 | 0.000 |
| checkpoint_000050_seed_1009_black-purple_purple | 0.000 | 0.003 | 0.000 | 1.200 | 0.000 |
| checkpoint_000050_seed_1000 | 0.001 | 0.001 | 0.001 | 1.108 | 0.000 |
| checkpoint_000050_seed_1001 | 0.000 | 0.001 | 0.001 | 1.189 | 0.000 |
| checkpoint_000050_seed_1002 | 0.000 | 0.001 | 0.001 | 1.179 | 0.000 |
| checkpoint_000050_seed_1003 | 0.001 | 0.001 | 0.001 | 1.189 | 0.000 |
| checkpoint_000050_seed_1004 | 0.000 | 0.001 | 0.001 | 1.154 | 0.000 |
| checkpoint_000050_seed_1005 | 0.000 | 0.001 | 0.001 | 1.182 | 0.000 |
| checkpoint_000050_seed_1006 | 0.000 | 0.001 | 0.001 | 1.184 | 0.000 |
| checkpoint_000050_seed_1007 | 0.000 | 0.001 | 0.001 | 1.179 | 0.000 |
| checkpoint_000050_seed_1008 | 0.001 | 0.001 | 0.001 | 1.119 | 0.000 |
| checkpoint_000050_seed_1009 | 0.000 | 0.001 | 0.001 | 1.147 | 0.000 |
| checkpoint_000050_seed_1010 | 0.001 | 0.001 | 0.001 | 1.148 | 0.000 |
| checkpoint_000050_seed_1011 | 0.000 | 0.001 | 0.001 | 1.157 | 0.000 |
| checkpoint_000050_seed_1012 | 0.000 | 0.001 | 0.001 | 1.150 | 0.000 |
| checkpoint_000050_seed_1013 | 0.000 | 0.001 | 0.001 | 1.091 | 0.000 |
| checkpoint_000050_seed_1014 | 0.000 | 0.001 | 0.001 | 1.134 | 0.000 |
| checkpoint_000050_seed_1015 | 0.000 | 0.001 | 0.001 | 1.104 | 0.000 |
| checkpoint_000050_seed_1016 | 0.000 | 0.001 | 0.001 | 1.112 | 0.000 |
| checkpoint_000050_seed_1017 | 0.001 | 0.001 | 0.001 | 1.133 | 0.000 |
| checkpoint_000050_seed_1018 | 0.000 | 0.001 | 0.001 | 1.100 | 0.000 |
| checkpoint_000050_seed_1019 | 0.000 | 0.001 | 0.001 | 1.120 | 0.000 |
| checkpoint_000100_seed_1000 | 0.000 | 0.001 | 0.001 | 1.515 | 0.000 |
| checkpoint_000100_seed_1001 | 0.000 | 0.001 | 0.001 | 1.207 | 0.000 |
| checkpoint_000100_seed_1002 | 0.000 | 0.001 | 0.001 | 1.217 | 0.000 |
| checkpoint_000100_seed_1003 | 0.000 | 0.001 | 0.001 | 1.217 | 0.000 |
| checkpoint_000100_seed_1004 | 0.000 | 0.001 | 0.001 | 1.227 | 0.000 |
| checkpoint_000100_seed_1005 | 0.001 | 0.001 | 0.001 | 1.252 | 0.000 |
| checkpoint_000100_seed_1006 | 0.000 | 0.001 | 0.001 | 1.192 | 0.000 |
| checkpoint_000100_seed_1007 | 0.001 | 0.001 | 0.001 | 1.163 | 0.000 |
| checkpoint_000100_seed_1008 | 0.000 | 0.001 | 0.001 | 1.137 | 0.000 |
| checkpoint_000100_seed_1009 | 0.000 | 0.001 | 0.001 | 1.144 | 0.000 |
| checkpoint_000100_seed_1010 | 0.000 | 0.001 | 0.001 | 1.212 | 0.000 |
| checkpoint_000100_seed_1011 | 0.000 | 0.001 | 0.001 | 1.124 | 0.000 |
| checkpoint_000100_seed_1012 | 0.000 | 0.001 | 0.001 | 1.169 | 0.000 |
| checkpoint_000100_seed_1013 | 0.000 | 0.001 | 0.001 | 1.218 | 0.000 |
| checkpoint_000100_seed_1014 | 0.000 | 0.001 | 0.001 | 1.207 | 0.000 |
| checkpoint_000100_seed_1015 | 0.001 | 0.001 | 0.001 | 1.243 | 0.000 |
| checkpoint_000100_seed_1016 | 0.000 | 0.001 | 0.001 | 1.130 | 0.000 |
| checkpoint_000100_seed_1017 | 0.000 | 0.001 | 0.001 | 2.047 | 0.000 |
| checkpoint_000100_seed_1018 | 0.000 | 0.001 | 0.001 | 1.225 | 0.000 |
| checkpoint_000100_seed_1019 | 0.001 | 0.001 | 0.001 | 1.388 | 0.000 |
| checkpoint_000300_seed_1000 | 0.000 | 0.001 | 0.001 | 1.160 | 0.027 |
| checkpoint_000300_seed_1001 | 0.000 | 0.001 | 0.001 | 1.143 | 0.000 |
| checkpoint_000300_seed_1002 | 0.000 | 0.001 | 0.001 | 1.103 | 0.000 |
| checkpoint_000300_seed_1003 | 0.000 | 0.001 | 0.001 | 1.179 | 0.000 |
| checkpoint_000300_seed_1004 | 0.000 | 0.001 | 0.001 | 1.180 | 0.000 |
| checkpoint_000300_seed_1005 | 0.000 | 0.001 | 0.001 | 1.197 | 0.000 |
| checkpoint_000300_seed_1006 | 0.001 | 0.001 | 0.000 | 1.193 | 0.000 |
| checkpoint_000300_seed_1007 | 0.001 | 0.001 | 0.000 | 1.175 | 0.000 |
| checkpoint_000300_seed_1008 | 0.000 | 0.001 | 0.000 | 1.157 | 0.000 |
| checkpoint_000300_seed_1009 | 0.000 | 0.001 | 0.001 | 1.196 | 0.000 |
| checkpoint_000300_seed_1010 | 0.001 | 0.001 | 0.000 | 1.175 | 0.000 |
| checkpoint_000300_seed_1011 | 0.000 | 0.001 | 0.001 | 1.267 | 0.000 |
| checkpoint_000300_seed_1012 | 0.001 | 0.001 | 0.000 | 1.160 | 0.001 |
| checkpoint_000300_seed_1013 | 0.001 | 0.001 | 0.001 | 1.173 | 0.000 |
| checkpoint_000300_seed_1014 | 0.001 | 0.001 | 0.000 | 1.166 | 0.000 |
| checkpoint_000300_seed_1015 | 0.001 | 0.001 | 0.000 | 1.257 | 0.000 |
| checkpoint_000300_seed_1016 | 0.001 | 0.001 | 0.000 | 1.177 | 0.000 |
| checkpoint_000300_seed_1017 | 0.001 | 0.001 | 0.000 | 1.207 | 0.000 |
| checkpoint_000300_seed_1018 | 0.001 | 0.001 | 0.001 | 1.278 | 0.000 |
| checkpoint_000300_seed_1019 | 0.001 | 0.001 | 0.001 | 1.251 | 0.000 |
| checkpoint_000500_seed_1000 | 0.000 | 0.001 | 0.001 | 1.195 | 0.021 |
| checkpoint_000500_seed_1001 | 0.000 | 0.001 | 0.001 | 1.258 | 0.000 |
| checkpoint_000500_seed_1002 | 0.000 | 0.001 | 0.001 | 1.224 | 0.000 |
| checkpoint_000500_seed_1003 | 0.000 | 0.001 | 0.001 | 1.269 | 0.000 |
| checkpoint_000500_seed_1004 | 0.000 | 0.001 | 0.001 | 1.159 | 0.000 |
| checkpoint_000500_seed_1005 | 0.000 | 0.001 | 0.001 | 1.182 | 0.000 |
| checkpoint_000500_seed_1006 | 0.000 | 0.001 | 0.001 | 1.189 | 0.000 |
| checkpoint_000500_seed_1007 | 0.000 | 0.001 | 0.001 | 1.217 | 0.000 |
| checkpoint_000500_seed_1008 | 0.000 | 0.001 | 0.001 | 1.129 | 0.000 |
| checkpoint_000500_seed_1009 | 0.000 | 0.001 | 0.001 | 1.143 | 0.000 |
| checkpoint_000500_seed_1010 | 0.000 | 0.001 | 0.001 | 1.124 | 0.000 |
| checkpoint_000500_seed_1011 | 0.000 | 0.001 | 0.001 | 1.125 | 0.000 |
| checkpoint_000500_seed_1012 | 0.000 | 0.001 | 0.001 | 1.100 | 0.000 |
| checkpoint_000500_seed_1013 | 0.000 | 0.001 | 0.001 | 1.112 | 0.000 |
| checkpoint_000500_seed_1014 | 0.000 | 0.001 | 0.001 | 1.112 | 0.000 |
| checkpoint_000500_seed_1015 | 0.000 | 0.001 | 0.001 | 1.134 | 0.000 |
| checkpoint_000500_seed_1016 | 0.000 | 0.001 | 0.001 | 1.117 | 0.000 |
| checkpoint_000500_seed_1017 | 0.000 | 0.001 | 0.001 | 1.116 | 0.000 |
| checkpoint_000500_seed_1018 | 0.000 | 0.001 | 0.001 | 1.114 | 0.000 |
| checkpoint_000500_seed_1019 | 0.000 | 0.001 | 0.001 | 1.129 | 0.000 |
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
| checkpoint_000050_seed_1000_blue | 0.017 | 0.003 | 0.027 | 0.029 | 0.076 | 0.012 |
| checkpoint_000050_seed_1000_red | 0.017 | 0.002 | 0.022 | 0.033 | 0.074 | 0.013 |
| checkpoint_000050_seed_1001_blue | 0.019 | 0.003 | 0.028 | 0.032 | 0.081 | 0.013 |
| checkpoint_000050_seed_1001_red | 0.019 | 0.002 | 0.023 | 0.031 | 0.075 | 0.014 |
| checkpoint_000050_seed_1002_blue | 0.017 | 0.003 | 0.028 | 0.031 | 0.080 | 0.015 |
| checkpoint_000050_seed_1002_red | 0.018 | 0.003 | 0.028 | 0.033 | 0.082 | 0.015 |
| checkpoint_000050_seed_1003_blue | 0.017 | 0.003 | 0.025 | 0.034 | 0.079 | 0.011 |
| checkpoint_000050_seed_1003_red | 0.018 | 0.003 | 0.026 | 0.033 | 0.079 | 0.012 |
| checkpoint_000050_seed_1004_blue | 0.018 | 0.002 | 0.028 | 0.033 | 0.081 | 0.013 |
| checkpoint_000050_seed_1004_red | 0.016 | 0.003 | 0.028 | 0.031 | 0.079 | 0.014 |
| checkpoint_000050_seed_1005_blue | 0.019 | 0.002 | 0.027 | 0.037 | 0.086 | 0.013 |
| checkpoint_000050_seed_1005_red | 0.016 | 0.003 | 0.027 | 0.032 | 0.078 | 0.013 |
| checkpoint_000050_seed_1006_blue | 0.017 | 0.003 | 0.029 | 0.033 | 0.081 | 0.012 |
| checkpoint_000050_seed_1006_red | 0.018 | 0.002 | 0.025 | 0.031 | 0.076 | 0.012 |
| checkpoint_000050_seed_1007_blue | 0.017 | 0.001 | 0.017 | 0.032 | 0.068 | 0.009 |
| checkpoint_000050_seed_1007_red | 0.018 | 0.002 | 0.027 | 0.038 | 0.085 | 0.012 |
| checkpoint_000050_seed_1008_blue | 0.017 | 0.002 | 0.024 | 0.033 | 0.077 | 0.011 |
| checkpoint_000050_seed_1008_red | 0.019 | 0.002 | 0.025 | 0.033 | 0.079 | 0.011 |
| checkpoint_000050_seed_1009_blue | 0.017 | 0.002 | 0.026 | 0.034 | 0.079 | 0.010 |
| checkpoint_000050_seed_1009_red | 0.019 | 0.002 | 0.026 | 0.037 | 0.084 | 0.011 |
| checkpoint_000100_seed_1000_blue | 0.019 | 0.001 | 0.014 | 0.031 | 0.065 | 0.009 |
| checkpoint_000100_seed_1000_red | 0.017 | 0.003 | 0.026 | 0.032 | 0.078 | 0.013 |
| checkpoint_000100_seed_1001_blue | 0.017 | 0.001 | 0.012 | 0.032 | 0.061 | 0.008 |
| checkpoint_000100_seed_1001_red | 0.018 | 0.001 | 0.013 | 0.036 | 0.068 | 0.010 |
| checkpoint_000100_seed_1002_blue | 0.018 | 0.002 | 0.023 | 0.030 | 0.074 | 0.012 |
| checkpoint_000100_seed_1002_red | 0.018 | 0.002 | 0.022 | 0.034 | 0.076 | 0.011 |
| checkpoint_000100_seed_1003_blue | 0.018 | 0.002 | 0.025 | 0.033 | 0.078 | 0.014 |
| checkpoint_000100_seed_1003_red | 0.017 | 0.002 | 0.019 | 0.032 | 0.070 | 0.012 |
| checkpoint_000100_seed_1004_blue | 0.018 | 0.002 | 0.024 | 0.030 | 0.075 | 0.013 |
| checkpoint_000100_seed_1004_red | 0.016 | 0.003 | 0.026 | 0.031 | 0.076 | 0.015 |
| checkpoint_000100_seed_1005_blue | 0.018 | 0.002 | 0.025 | 0.029 | 0.074 | 0.012 |
| checkpoint_000100_seed_1005_red | 0.017 | 0.002 | 0.025 | 0.026 | 0.072 | 0.012 |
| checkpoint_000100_seed_1006_blue | 0.017 | 0.003 | 0.025 | 0.026 | 0.071 | 0.013 |
| checkpoint_000100_seed_1006_red | 0.018 | 0.003 | 0.027 | 0.034 | 0.081 | 0.013 |
| checkpoint_000100_seed_1007_blue | 0.017 | 0.003 | 0.026 | 0.027 | 0.073 | 0.014 |
| checkpoint_000100_seed_1007_red | 0.017 | 0.003 | 0.025 | 0.028 | 0.072 | 0.015 |
| checkpoint_000100_seed_1008_blue | 0.017 | 0.003 | 0.025 | 0.027 | 0.072 | 0.014 |
| checkpoint_000100_seed_1008_red | 0.017 | 0.003 | 0.026 | 0.029 | 0.075 | 0.013 |
| checkpoint_000100_seed_1009_blue | 0.017 | 0.003 | 0.026 | 0.026 | 0.072 | 0.012 |
| checkpoint_000100_seed_1009_red | 0.017 | 0.002 | 0.025 | 0.025 | 0.070 | 0.015 |
| checkpoint_000300_seed_1000_blue | 0.018 | 0.002 | 0.021 | 0.031 | 0.072 | 0.011 |
| checkpoint_000300_seed_1000_red | 0.019 | 0.001 | 0.015 | 0.034 | 0.069 | 0.011 |
| checkpoint_000300_seed_1001_blue | 0.018 | 0.003 | 0.026 | 0.029 | 0.075 | 0.013 |
| checkpoint_000300_seed_1001_red | 0.018 | 0.003 | 0.028 | 0.032 | 0.080 | 0.012 |
| checkpoint_000300_seed_1002_blue | 0.018 | 0.003 | 0.028 | 0.032 | 0.080 | 0.014 |
| checkpoint_000300_seed_1002_red | 0.018 | 0.002 | 0.027 | 0.031 | 0.079 | 0.012 |
| checkpoint_000300_seed_1003_blue | 0.018 | 0.003 | 0.027 | 0.030 | 0.078 | 0.013 |
| checkpoint_000300_seed_1003_red | 0.018 | 0.003 | 0.027 | 0.033 | 0.081 | 0.013 |
| checkpoint_000300_seed_1004_blue | 0.018 | 0.003 | 0.027 | 0.032 | 0.080 | 0.013 |
| checkpoint_000300_seed_1004_red | 0.018 | 0.003 | 0.028 | 0.032 | 0.080 | 0.013 |
| checkpoint_000300_seed_1005_blue | 0.018 | 0.002 | 0.022 | 0.034 | 0.075 | 0.013 |
| checkpoint_000300_seed_1005_red | 0.017 | 0.003 | 0.026 | 0.033 | 0.079 | 0.013 |
| checkpoint_000300_seed_1006_blue | 0.017 | 0.002 | 0.026 | 0.031 | 0.077 | 0.011 |
| checkpoint_000300_seed_1006_red | 0.018 | 0.001 | 0.016 | 0.029 | 0.064 | 0.009 |
| checkpoint_000300_seed_1007_blue | 0.017 | 0.002 | 0.024 | 0.029 | 0.072 | 0.013 |
| checkpoint_000300_seed_1007_red | 0.017 | 0.003 | 0.025 | 0.031 | 0.076 | 0.014 |
| checkpoint_000300_seed_1008_blue | 0.016 | 0.003 | 0.026 | 0.031 | 0.076 | 0.013 |
| checkpoint_000300_seed_1008_red | 0.016 | 0.002 | 0.025 | 0.031 | 0.075 | 0.011 |
| checkpoint_000300_seed_1009_blue | 0.016 | 0.002 | 0.025 | 0.032 | 0.075 | 0.013 |
| checkpoint_000300_seed_1009_red | 0.017 | 0.002 | 0.025 | 0.032 | 0.075 | 0.011 |
| checkpoint_000500_seed_1000_blue | 0.017 | 0.002 | 0.021 | 0.033 | 0.073 | 0.012 |
| checkpoint_000500_seed_1000_red | 0.017 | 0.003 | 0.026 | 0.030 | 0.075 | 0.012 |
| checkpoint_000500_seed_1001_blue | 0.018 | 0.001 | 0.012 | 0.027 | 0.057 | 0.009 |
| checkpoint_000500_seed_1001_red | 0.017 | 0.002 | 0.019 | 0.031 | 0.068 | 0.012 |
| checkpoint_000500_seed_1002_blue | 0.017 | 0.003 | 0.026 | 0.030 | 0.076 | 0.013 |
| checkpoint_000500_seed_1002_red | 0.018 | 0.002 | 0.025 | 0.033 | 0.078 | 0.013 |
| checkpoint_000500_seed_1003_blue | 0.017 | 0.003 | 0.027 | 0.030 | 0.076 | 0.015 |
| checkpoint_000500_seed_1003_red | 0.017 | 0.003 | 0.025 | 0.030 | 0.074 | 0.013 |
| checkpoint_000500_seed_1004_blue | 0.017 | 0.003 | 0.026 | 0.029 | 0.076 | 0.015 |
| checkpoint_000500_seed_1004_red | 0.017 | 0.003 | 0.026 | 0.030 | 0.075 | 0.014 |
| checkpoint_000500_seed_1005_blue | 0.018 | 0.002 | 0.020 | 0.031 | 0.070 | 0.012 |
| checkpoint_000500_seed_1005_red | 0.017 | 0.003 | 0.026 | 0.031 | 0.076 | 0.015 |
| checkpoint_000500_seed_1006_blue | 0.016 | 0.002 | 0.025 | 0.027 | 0.071 | 0.015 |
| checkpoint_000500_seed_1006_red | 0.017 | 0.002 | 0.026 | 0.027 | 0.072 | 0.014 |
| checkpoint_000500_seed_1007_blue | 0.017 | 0.003 | 0.026 | 0.026 | 0.072 | 0.014 |
| checkpoint_000500_seed_1007_red | 0.017 | 0.002 | 0.026 | 0.026 | 0.071 | 0.013 |
| checkpoint_000500_seed_1008_blue | 0.017 | 0.003 | 0.026 | 0.026 | 0.071 | 0.013 |
| checkpoint_000500_seed_1008_red | 0.017 | 0.002 | 0.025 | 0.027 | 0.070 | 0.011 |
| checkpoint_000500_seed_1009_blue | 0.016 | 0.003 | 0.026 | 0.026 | 0.071 | 0.013 |
| checkpoint_000500_seed_1009_red | 0.017 | 0.003 | 0.027 | 0.027 | 0.074 | 0.015 |
| checkpoint_000050_seed_1000_black-purple_black | 0.017 | 0.002 | 0.023 | 0.028 | 0.070 | 0.014 |
| checkpoint_000050_seed_1000_black-purple_purple | 0.017 | 0.002 | 0.024 | 0.030 | 0.074 | 0.017 |
| checkpoint_000050_seed_1000_green-yellow_green | 0.015 | 0.001 | 0.011 | 0.020 | 0.047 | 0.011 |
| checkpoint_000050_seed_1000_green-yellow_yellow | 0.015 | 0.001 | 0.012 | 0.022 | 0.050 | 0.012 |
| checkpoint_000050_seed_1001_black-purple_black | 0.017 | 0.002 | 0.022 | 0.031 | 0.072 | 0.013 |
| checkpoint_000050_seed_1001_black-purple_purple | 0.017 | 0.002 | 0.021 | 0.032 | 0.071 | 0.014 |
| checkpoint_000050_seed_1001_green-yellow_green | 0.015 | 0.001 | 0.011 | 0.023 | 0.050 | 0.014 |
| checkpoint_000050_seed_1001_green-yellow_yellow | 0.015 | 0.001 | 0.011 | 0.022 | 0.049 | 0.011 |
| checkpoint_000050_seed_1002_black-purple_black | 0.018 | 0.001 | 0.017 | 0.030 | 0.066 | 0.012 |
| checkpoint_000050_seed_1002_black-purple_purple | 0.017 | 0.002 | 0.019 | 0.030 | 0.067 | 0.013 |
| checkpoint_000050_seed_1002_green-yellow_green | 0.015 | 0.001 | 0.011 | 0.023 | 0.050 | 0.013 |
| checkpoint_000050_seed_1002_green-yellow_yellow | 0.016 | 0.001 | 0.011 | 0.022 | 0.050 | 0.012 |
| checkpoint_000050_seed_1003_black-purple_black | 0.016 | 0.003 | 0.026 | 0.032 | 0.077 | 0.019 |
| checkpoint_000050_seed_1003_black-purple_purple | 0.017 | 0.001 | 0.017 | 0.029 | 0.063 | 0.014 |
| checkpoint_000050_seed_1003_green-yellow_green | 0.017 | 0.001 | 0.018 | 0.028 | 0.063 | 0.012 |
| checkpoint_000050_seed_1003_green-yellow_yellow | 0.018 | 0.002 | 0.019 | 0.027 | 0.065 | 0.015 |
| checkpoint_000050_seed_1004_black-purple_black | 0.017 | 0.002 | 0.026 | 0.030 | 0.074 | 0.016 |
| checkpoint_000050_seed_1004_black-purple_purple | 0.017 | 0.002 | 0.024 | 0.027 | 0.070 | 0.016 |
| checkpoint_000050_seed_1004_green-yellow_green | 0.017 | 0.002 | 0.019 | 0.027 | 0.064 | 0.014 |
| checkpoint_000050_seed_1004_green-yellow_yellow | 0.016 | 0.001 | 0.013 | 0.025 | 0.055 | 0.012 |
| checkpoint_000050_seed_1005_black-purple_black | 0.017 | 0.002 | 0.024 | 0.028 | 0.071 | 0.017 |
| checkpoint_000050_seed_1005_black-purple_purple | 0.017 | 0.003 | 0.027 | 0.031 | 0.077 | 0.018 |
| checkpoint_000050_seed_1006_black-purple_black | 0.017 | 0.002 | 0.023 | 0.027 | 0.069 | 0.016 |
| checkpoint_000050_seed_1006_black-purple_purple | 0.017 | 0.002 | 0.024 | 0.027 | 0.071 | 0.016 |
| checkpoint_000050_seed_1007_black-purple_black | 0.017 | 0.002 | 0.022 | 0.029 | 0.070 | 0.015 |
| checkpoint_000050_seed_1007_black-purple_purple | 0.017 | 0.002 | 0.025 | 0.029 | 0.073 | 0.015 |
| checkpoint_000050_seed_1008_black-purple_black | 0.017 | 0.002 | 0.024 | 0.028 | 0.070 | 0.016 |
| checkpoint_000050_seed_1008_black-purple_purple | 0.018 | 0.002 | 0.019 | 0.027 | 0.065 | 0.013 |
| checkpoint_000050_seed_1009_black-purple_black | 0.016 | 0.001 | 0.012 | 0.021 | 0.050 | 0.011 |
| checkpoint_000050_seed_1009_black-purple_purple | 0.015 | 0.001 | 0.011 | 0.023 | 0.050 | 0.011 |
| checkpoint_000050_seed_1000 | 0.017 | 0.003 | 0.024 | 0.027 | 0.070 | 0.015 |
| checkpoint_000050_seed_1001 | 0.017 | 0.002 | 0.022 | 0.028 | 0.069 | 0.013 |
| checkpoint_000050_seed_1002 | 0.017 | 0.003 | 0.024 | 0.030 | 0.073 | 0.014 |
| checkpoint_000050_seed_1003 | 0.017 | 0.003 | 0.026 | 0.029 | 0.075 | 0.011 |
| checkpoint_000050_seed_1004 | 0.017 | 0.003 | 0.024 | 0.027 | 0.071 | 0.014 |
| checkpoint_000050_seed_1005 | 0.017 | 0.003 | 0.025 | 0.028 | 0.072 | 0.014 |
| checkpoint_000050_seed_1006 | 0.017 | 0.002 | 0.022 | 0.025 | 0.066 | 0.013 |
| checkpoint_000050_seed_1007 | 0.017 | 0.002 | 0.024 | 0.026 | 0.069 | 0.013 |
| checkpoint_000050_seed_1008 | 0.017 | 0.002 | 0.025 | 0.026 | 0.071 | 0.013 |
| checkpoint_000050_seed_1009 | 0.017 | 0.003 | 0.025 | 0.027 | 0.072 | 0.014 |
| checkpoint_000050_seed_1010 | 0.018 | 0.003 | 0.024 | 0.026 | 0.071 | 0.015 |
| checkpoint_000050_seed_1011 | 0.017 | 0.003 | 0.023 | 0.029 | 0.071 | 0.013 |
| checkpoint_000050_seed_1012 | 0.017 | 0.002 | 0.024 | 0.027 | 0.070 | 0.014 |
| checkpoint_000050_seed_1013 | 0.017 | 0.003 | 0.024 | 0.026 | 0.070 | 0.014 |
| checkpoint_000050_seed_1014 | 0.017 | 0.003 | 0.024 | 0.027 | 0.071 | 0.015 |
| checkpoint_000050_seed_1015 | 0.017 | 0.002 | 0.023 | 0.026 | 0.069 | 0.015 |
| checkpoint_000050_seed_1016 | 0.017 | 0.002 | 0.024 | 0.027 | 0.071 | 0.014 |
| checkpoint_000050_seed_1017 | 0.017 | 0.002 | 0.024 | 0.026 | 0.070 | 0.015 |
| checkpoint_000050_seed_1018 | 0.017 | 0.002 | 0.024 | 0.025 | 0.068 | 0.015 |
| checkpoint_000050_seed_1019 | 0.017 | 0.002 | 0.024 | 0.024 | 0.067 | 0.013 |
| checkpoint_000100_seed_1000 | 0.017 | 0.002 | 0.024 | 0.028 | 0.071 | 0.014 |
| checkpoint_000100_seed_1001 | 0.016 | 0.002 | 0.022 | 0.027 | 0.068 | 0.011 |
| checkpoint_000100_seed_1002 | 0.017 | 0.002 | 0.024 | 0.028 | 0.072 | 0.013 |
| checkpoint_000100_seed_1003 | 0.017 | 0.002 | 0.023 | 0.026 | 0.068 | 0.014 |
| checkpoint_000100_seed_1004 | 0.017 | 0.002 | 0.022 | 0.027 | 0.068 | 0.013 |
| checkpoint_000100_seed_1005 | 0.018 | 0.003 | 0.025 | 0.026 | 0.071 | 0.014 |
| checkpoint_000100_seed_1006 | 0.017 | 0.002 | 0.024 | 0.027 | 0.070 | 0.012 |
| checkpoint_000100_seed_1007 | 0.017 | 0.002 | 0.024 | 0.026 | 0.070 | 0.014 |
| checkpoint_000100_seed_1008 | 0.017 | 0.002 | 0.024 | 0.027 | 0.070 | 0.015 |
| checkpoint_000100_seed_1009 | 0.018 | 0.002 | 0.024 | 0.030 | 0.073 | 0.012 |
| checkpoint_000100_seed_1010 | 0.017 | 0.003 | 0.025 | 0.027 | 0.072 | 0.014 |
| checkpoint_000100_seed_1011 | 0.017 | 0.003 | 0.025 | 0.027 | 0.071 | 0.013 |
| checkpoint_000100_seed_1012 | 0.017 | 0.002 | 0.023 | 0.027 | 0.070 | 0.014 |
| checkpoint_000100_seed_1013 | 0.017 | 0.003 | 0.025 | 0.027 | 0.072 | 0.015 |
| checkpoint_000100_seed_1014 | 0.017 | 0.002 | 0.023 | 0.026 | 0.069 | 0.012 |
| checkpoint_000100_seed_1015 | 0.017 | 0.002 | 0.024 | 0.028 | 0.071 | 0.013 |
| checkpoint_000100_seed_1016 | 0.017 | 0.003 | 0.024 | 0.028 | 0.071 | 0.015 |
| checkpoint_000100_seed_1017 | 0.017 | 0.003 | 0.025 | 0.031 | 0.075 | 0.015 |
| checkpoint_000100_seed_1018 | 0.017 | 0.002 | 0.023 | 0.028 | 0.070 | 0.013 |
| checkpoint_000100_seed_1019 | 0.017 | 0.003 | 0.023 | 0.031 | 0.073 | 0.014 |
| checkpoint_000300_seed_1000 | 0.017 | 0.003 | 0.024 | 0.026 | 0.069 | 0.013 |
| checkpoint_000300_seed_1001 | 0.017 | 0.003 | 0.024 | 0.026 | 0.070 | 0.013 |
| checkpoint_000300_seed_1002 | 0.017 | 0.003 | 0.024 | 0.025 | 0.069 | 0.015 |
| checkpoint_000300_seed_1003 | 0.017 | 0.003 | 0.024 | 0.027 | 0.071 | 0.013 |
| checkpoint_000300_seed_1004 | 0.017 | 0.003 | 0.024 | 0.028 | 0.071 | 0.014 |
| checkpoint_000300_seed_1005 | 0.017 | 0.002 | 0.025 | 0.031 | 0.075 | 0.014 |
| checkpoint_000300_seed_1006 | 0.019 | 0.002 | 0.025 | 0.040 | 0.086 | 0.011 |
| checkpoint_000300_seed_1007 | 0.018 | 0.002 | 0.026 | 0.034 | 0.080 | 0.013 |
| checkpoint_000300_seed_1008 | 0.018 | 0.002 | 0.027 | 0.032 | 0.080 | 0.013 |
| checkpoint_000300_seed_1009 | 0.018 | 0.003 | 0.026 | 0.033 | 0.079 | 0.016 |
| checkpoint_000300_seed_1010 | 0.018 | 0.002 | 0.026 | 0.032 | 0.078 | 0.012 |
| checkpoint_000300_seed_1011 | 0.017 | 0.002 | 0.024 | 0.032 | 0.076 | 0.012 |
| checkpoint_000300_seed_1012 | 0.018 | 0.002 | 0.024 | 0.034 | 0.079 | 0.013 |
| checkpoint_000300_seed_1013 | 0.018 | 0.002 | 0.025 | 0.031 | 0.077 | 0.012 |
| checkpoint_000300_seed_1014 | 0.019 | 0.002 | 0.027 | 0.035 | 0.084 | 0.014 |
| checkpoint_000300_seed_1015 | 0.019 | 0.002 | 0.027 | 0.033 | 0.082 | 0.013 |
| checkpoint_000300_seed_1016 | 0.019 | 0.002 | 0.027 | 0.036 | 0.085 | 0.013 |
| checkpoint_000300_seed_1017 | 0.019 | 0.002 | 0.026 | 0.036 | 0.083 | 0.011 |
| checkpoint_000300_seed_1018 | 0.020 | 0.002 | 0.027 | 0.037 | 0.086 | 0.011 |
| checkpoint_000300_seed_1019 | 0.020 | 0.002 | 0.026 | 0.036 | 0.084 | 0.012 |
| checkpoint_000500_seed_1000 | 0.017 | 0.002 | 0.021 | 0.028 | 0.069 | 0.013 |
| checkpoint_000500_seed_1001 | 0.016 | 0.003 | 0.023 | 0.031 | 0.073 | 0.014 |
| checkpoint_000500_seed_1002 | 0.016 | 0.003 | 0.025 | 0.034 | 0.078 | 0.014 |
| checkpoint_000500_seed_1003 | 0.017 | 0.002 | 0.024 | 0.031 | 0.074 | 0.013 |
| checkpoint_000500_seed_1004 | 0.016 | 0.002 | 0.024 | 0.031 | 0.074 | 0.014 |
| checkpoint_000500_seed_1005 | 0.017 | 0.003 | 0.024 | 0.031 | 0.074 | 0.014 |
| checkpoint_000500_seed_1006 | 0.016 | 0.002 | 0.024 | 0.030 | 0.072 | 0.011 |
| checkpoint_000500_seed_1007 | 0.016 | 0.002 | 0.023 | 0.032 | 0.073 | 0.011 |
| checkpoint_000500_seed_1008 | 0.017 | 0.003 | 0.023 | 0.032 | 0.074 | 0.014 |
| checkpoint_000500_seed_1009 | 0.017 | 0.003 | 0.025 | 0.029 | 0.074 | 0.014 |
| checkpoint_000500_seed_1010 | 0.016 | 0.002 | 0.024 | 0.031 | 0.074 | 0.015 |
| checkpoint_000500_seed_1011 | 0.016 | 0.002 | 0.024 | 0.030 | 0.072 | 0.013 |
| checkpoint_000500_seed_1012 | 0.017 | 0.002 | 0.025 | 0.029 | 0.073 | 0.016 |
| checkpoint_000500_seed_1013 | 0.017 | 0.002 | 0.025 | 0.028 | 0.072 | 0.013 |
| checkpoint_000500_seed_1014 | 0.017 | 0.002 | 0.024 | 0.029 | 0.073 | 0.014 |
| checkpoint_000500_seed_1015 | 0.017 | 0.002 | 0.023 | 0.029 | 0.071 | 0.013 |
| checkpoint_000500_seed_1016 | 0.017 | 0.002 | 0.024 | 0.030 | 0.073 | 0.015 |
| checkpoint_000500_seed_1017 | 0.017 | 0.002 | 0.024 | 0.029 | 0.073 | 0.015 |
| checkpoint_000500_seed_1018 | 0.016 | 0.003 | 0.025 | 0.029 | 0.073 | 0.014 |
| checkpoint_000500_seed_1019 | 0.017 | 0.002 | 0.024 | 0.029 | 0.072 | 0.015 |
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
| checkpoint_000050_seed_1000_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1000_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1001_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1001_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1002_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1002_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1003_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1003_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1004_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1004_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1005_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1005_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1006_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1006_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1007_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1007_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1008_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1008_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1009_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1009_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.278 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1000_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1000_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1001_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1001_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1002_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1002_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1003_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1003_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1004_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1004_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1005_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1005_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1006_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1006_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1007_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1007_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1008_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1008_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1009_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1009_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000100 | 7.282 | 0.044 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1000_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1000_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1001_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1001_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1002_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1002_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1003_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1003_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1004_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1004_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1005_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1005_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1006_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1006_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1007_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1007_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1008_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1008_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1009_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1009_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 | 6.786 | 0.040 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1000_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1000_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1001_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1001_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1002_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1002_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1003_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1003_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1004_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1004_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1005_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1005_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1006_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1006_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1007_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1007_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1008_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1008_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1009_blue | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1009_red | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000500 | 6.285 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1000_black-purple_black | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1000_black-purple_purple | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1000_green-yellow_green | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1000_green-yellow_yellow | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1001_black-purple_black | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1001_black-purple_purple | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1001_green-yellow_green | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1001_green-yellow_yellow | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1002_black-purple_black | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1002_black-purple_purple | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1002_green-yellow_green | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1002_green-yellow_yellow | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1003_black-purple_black | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1003_black-purple_purple | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1003_green-yellow_green | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1003_green-yellow_yellow | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1004_black-purple_black | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1004_black-purple_purple | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1004_green-yellow_green | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1004_green-yellow_yellow | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1005_black-purple_black | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1005_black-purple_purple | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1006_black-purple_black | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1006_black-purple_purple | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1007_black-purple_black | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1007_black-purple_purple | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1008_black-purple_black | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1008_black-purple_purple | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1009_black-purple_black | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1009_black-purple_purple | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000050 | 7.331 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1000 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1001 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1002 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1003 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1004 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1005 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1006 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1007 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1008 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1009 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1010 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1011 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1012 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1013 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1014 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1015 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1016 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1017 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1018 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000050_seed_1019 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000050 | 5.836 | 0.043 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1000 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1001 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1002 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1003 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1004 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1005 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1006 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1007 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1008 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1009 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1010 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1011 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1012 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1013 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1014 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1015 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1016 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1017 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1018 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000100_seed_1019 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000100 | 6.102 | 0.041 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1000 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1001 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1002 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1003 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1004 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1005 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1006 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1007 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1008 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1009 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1010 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1011 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1012 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1013 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1014 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1015 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1016 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1017 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1018 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000300_seed_1019 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300 | 5.055 | 0.042 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1000 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1001 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1002 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1003 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1004 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1005 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1006 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1007 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1008 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1009 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1010 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1011 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1012 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1013 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1014 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1015 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1016 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1017 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1018 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| checkpoint_000500_seed_1019 | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000500 | 8.017 | 0.046 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| 20261012_040655_255759 | 47.406 | Disabled | /workspace/openvla-7b | 52.546 | — | Not recorded | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| 20261012_043822_113279 | 23.547 | Disabled | /workspace/openvla-7b | 46.298 | — | Not recorded | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |

Model loading includes processor setup, checkpoint reading, construction and GPU transfer; it does not isolate storage-to-VRAM time. Historical ECC is not inferred from unrelated training pods.

## Refreshing this report

New local simulator trials refresh this report automatically after saving their run report. No cloud connections or predictions are made by report generation.

```bash
.venv/bin/python -m analysis.summarize_gpu_inference
```

Hourly rates are stored in `inference/gpu_inference_rates.json`. Raw measurements remain in `outputs/<run>/startup_info.json` and `step_*/execution.json`. Each run also keeps `RUN_REPORT.md`.
