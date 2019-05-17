============
FAQ
============

How does Foolbox handle inputs that are misclassified without any perturbation?
  The attacks will not be run and instead the unperturbed input is returned as an *adversarial* with distance 0 to the clean input.

What happens if an attack fails?
  The attack will return `None` and the distance will be `np.inf`.

Why is the returned adversarial not misclassified by my model?
  Most likely you have a discrepancy between how you evaluate your model and how you told Foolbox to evaluate it. For example, you might not be using the same preprocessing. Compare the output of the `predictions` method of the Foolbox model instance with your model's output (logits). This problem can also be caused by non-deterministic models. Make sure that your model is not stochastic and always returns the same output when given the same input. In rare cases it can also be that a seemlingly deterministic model becomes numerically stochastic around the decision boundary (e.g. because of non-deterministic floating point `reduce_sum` operations). You can always check `adversarial.output` and `adversarial.adversarial_class` to see the output Foolbox got from your model when deciding that this was an adversarial.

Why are the gradients multiplied by the bounds (`max_ - min_`)?
  This scaling is meant to make hyperparameters such as the `epsilon` for FGSM independent of the bounds. `epsilon = 0.1` thus means that you perturb the input by 10% relative to the `max - max` range (which could for example go from 0 to 1 or from 0 to 255).
