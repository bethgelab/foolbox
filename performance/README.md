## Performance comparison between Foolbox versions

|                                        |   Foolbox `1.8.0`   |   Foolbox `2.4.0`  | Foolbox `3.1.1`<br>(aka Native) |
|----------------------------------------|:-------------------:|:------------------:|:-------------------------------:|
| accuracy (single image)                |  `5.02 ms ± 338 µs` | `4.99 ms ± 378 µs` |      **`3.99 ms ± 131 µs`**     |
| accuracy (16 images)                   | `88.9 ms ± 8.24 ms` |  `12 ms ± 1.34 ms` |     **`8.21 ms ± 54.4 µs`**     |
| PGD attack (16 images, single epsilon) |      `161.8 s`      |      `37.5 s`      |           **`1.1 s`**           |
| PGD attack (16 images, 8 epsilons)     |      `164.6 s`      |      `36.9 s`      |           **`9.0 s`**           |


All experiments were done on an Nvidia GeForce GTX 1080 using the PGD attack.

Note that Foolbox 3 is faster because **1)** it avoids memory copies between GPU
and CPU by using EagerPy instead of NumPy, **2)** it fully supports batches
of inputs, and **3)** it currently uses a different approach for fixed-epsilon attacks
like PGD (instead of minimizing the perturbation, the attack is run for
each epsilon; this is more inline with what is generally expected;
for these attacks the duration therefore now scales with the
number of epsilons; it is however still faster and it produces better results).
