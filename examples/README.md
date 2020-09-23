## Examples

This folder contains examples that demonstrate how Foolbox can be used
to run one or more adversarial attacks and how to use the returned results
to compute the robust accuracy (the accuracy of the model when it is attacked).

The standard example can be found in:
* `single_attack_pytorch_resnet18.py`
* `single_attack_tensorflow_resnet50.py`

It shows how to run a single adversarial attack (Linf PGD) against an ImageNet
model in PyTorch and TensorFlow.

The remaining examples are all for PyTorch,
but the difference between these frameworks is really just replacing the model
at the beginning of the script. So any example can be easily run with any
framework.

`multiple_attacks_pytorch_resnet18.py` is an extended version of the single attack
example. It shows how to combine the results of running multiple attacks
to report the robust accuracy always using the strongest attack per sample.

`spatial_attack_pytorch_resnet18.py` shows how to use the Spatial Attack. This attack
is a bit special because it doesn't use Lp balls and instead considers translations
and rotations. It therefore has a custom example. All the other attacks can be
used like Linf PGD in the other examples above.

`substituion_model_pytorch_resnet18.py` shows how to replace the gradient of
a model with the gradient of another model. This can be useful when the original
model has bad gradients ("gradient masking", "obfuscated gradients").

The `zoo` folder shows how a model can be shared in a Foolbox Model Zoo compatible way.
