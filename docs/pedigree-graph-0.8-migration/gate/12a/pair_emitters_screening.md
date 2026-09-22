| fixture | receiver | threads | emitter | pairs | wall median (s) | wall range | engine RSS median (MiB) | engine RSS max | process peak (MiB) |
|---|---|---|---|---|---|---|---|---|---|
| random_300k | graph | 1 | bounded_wave | 30,820,305 | 16.028 | 16.028 to 16.727 | 327.8 | 327.9 | 338.6 |
| random_300k | graph | 1 | buffered | 30,820,305 | 16.569 | 16.479 to 17.434 | 528.9 | 528.9 | 539.6 |
| random_300k | graph | 1 | two_pass | 30,820,305 | 31.336 | 31.007 to 31.861 | 261.2 | 261.3 | 271.9 |
| random_300k | graph | 6 | bounded_wave | 30,820,305 | 4.323 | 4.256 to 4.523 | 410.3 | 422.4 | 421.0 |
| random_300k | graph | 6 | buffered | 30,820,305 | 3.941 | 3.913 to 4.006 | 537.4 | 538.4 | 548.1 |
| random_300k | graph | 6 | two_pass | 30,820,305 | 7.174 | 7.037 to 7.261 | 264.6 | 264.8 | 275.3 |
| random_30k | graph | 1 | bounded_wave | 3,067,768 | 1.285 | 1.260 to 1.361 | 37.8 | 37.8 | 41.1 |
| random_30k | graph | 1 | buffered | 3,067,768 | 1.291 | 1.270 to 1.430 | 49.5 | 49.6 | 52.8 |
| random_30k | graph | 1 | two_pass | 3,067,768 | 2.493 | 2.438 to 2.503 | 26.1 | 26.1 | 29.4 |
| random_30k | graph | 6 | bounded_wave | 3,067,768 | 0.320 | 0.315 to 0.356 | 45.2 | 45.8 | 48.5 |
| random_30k | graph | 6 | buffered | 3,067,768 | 0.325 | 0.324 to 0.344 | 50.8 | 53.6 | 54.0 |
| random_30k | graph | 6 | two_pass | 3,067,768 | 0.631 | 0.627 to 0.652 | 26.9 | 26.9 | 30.1 |
