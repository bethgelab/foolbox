---
title: 'Foolbox Native: Fast adversarial attacks to benchmark the robustness of machine learning models in PyTorch, TensorFlow, and JAX'
tags:
  - python
  - machine learning
  - adversarial attacks
  - neural networks
  - pytorch
  - tensorflow
  - jax
  - keras
  - eagerpy
authors:
  - name: Jonas Rauber
    orcid: 0000-0001-6795-9441
    affiliation: "1, 2"
  - name: Roland Zimmermann
    affiliation: "1, 2"
  - name: Matthias Bethge^[joint senior authors]
    affiliation: "1, 3"
  - name: Wieland Brendel
    affiliation: "1, 3"
affiliations:
 - name: Tübingen AI Center, University of Tübingen, Germany
   index: 1
 - name: International Max Planck Research School for Intelligent Systems, Tübingen, Germany
   index: 2
 - name: Bernstein Center for Computational Neuroscience Tübingen, Germany
   index: 3
date: 10 August 2020
bibliography: paper.bib
---

# Summary

Machine learning has made enormous progress in recent years and is now being used in many real-world applications. Nevertheless, even state-of-the-art machine learning models can be fooled by small, maliciously crafted perturbations of their input data. Foolbox is a popular Python library to benchmark the robustness of machine learning models against these adversarial perturbations. It comes with a huge collection of state-of-the-art adversarial attacks to find adversarial perturbations and thanks to its framework-agnostic design it is ideally suited for comparing the robustness of many different models implemented in different frameworks. Foolbox 3 aka Foolbox Native has been rewritten from scratch to achieve native performance on models developed in PyTorch [@pytorch], TensorFlow [@tensorflow], and JAX [@jax], all with one codebase without code duplication.

# Statement of need

Evaluating the adversarial robustness of machine learning models is crucial to understanding their shortcomings and quantifying the implications on safety, security, and interpretability. Foolbox Native is the first adversarial robustness toolbox that is both fast and framework-agnostic. This is important because modern machine learning models such as deep neural networks are often computationally expensive and are implemented in different frameworks such as PyTorch and TensorFlow. Foolbox Native combines the framework-agnostic design of the original Foolbox [@rauber2017foolbox] with real batch support and native performance in PyTorch, TensorFlow, and JAX, all using a single codebase without code duplication. To achieve this, all adversarial attacks have been rewritten from scratch and now use EagerPy [@rauber2020eagerpy] instead of NumPy [@numpy] to interface *natively* with the different frameworks.

This is great for both users and developers of adversarial attacks. Users can efficiently evaluate the robustness of different models in different frameworks using the same set of state-of-the-art adversarial attacks, thus obtaining comparable results. Attack developers do not need to choose between supporting just one framework or reimplementing their new adversarial attack multiple times and dealing with code duplication. In addition, they both benefit from the comprehensive type annotations [@pep484] in Foolbox Native to catch bugs even before running their code.

The combination of being framework-agnostic and simultaneously achieving native performance sets Foolbox Native apart from other adversarial attack libraries. The most popular alternative to Foolbox is CleverHans^[https://github.com/tensorflow/cleverhans]. It was the first adversarial attack library and has traditionally focused solely on TensorFlow (plans to make it framework-agnostic *in the future* have been announced). The original Foolbox was the second adversarial attack library and the first one to be framework-agnostic. Back then, this was achieved at the expense of performance. The adversarial robustness toolbox ART^[https://github.com/Trusted-AI/adversarial-robustness-toolbox] is another framework-agnostic adversarial attack library, but it is conceptually inspired by the original Foolbox and thus comes with the same performance trade-off. AdverTorch^[https://github.com/BorealisAI/advertorch] is a popular adversarial attack library that was inspired by the original Foolbox but improved its performance by focusing soley on PyTorch. Foolbox Native is our attempt to improve the performance of Foolbox without sacrificing the framework-agnostic design that is crucial to consistently evaluate the robustness of different machine learning models that use different frameworks.

# Use Cases

Foolbox was designed to make adversarial attacks easy to apply even without expert knowledge. It has been used in numerous scientific publications and has already been cited more than 220 times. On GitHub it has received contributions from several developers and has gathered more than 1.500 stars. It provides the reference implementations of various adversarial attacks, including the Boundary Attack [@brendel2018decisionbased], the Pointwise Attack [@schott2018towards], clipping-aware noise attacks [@rauber2020fast], the Brendel Bethge Attack [@brendel2019accurate], and the HopSkipJump Attack [@chen2020hopskipjumpattack], and is under active development since 2017.

# Acknowledgements

J.R. acknowledges support from the Bosch Research Foundation (Stifterverband, T113/30057/17) and the International Max Planck Research School for Intelligent Systems (IMPRS-IS). This work was supported by the German Federal Ministry of Education and Research (BMBF): Tübingen AI Center, FKZ: 01IS18039A, and by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/Interior Business Center (DoI/IBC) contract number D16PC00003. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DoI/IBC, or the U.S. Government.

We thank all contributors to Foolbox, in particular Behar Veliqi, Evgenia Rusak, Jianbo Chen, Rene Bidart, Jerome Rony, Ben Feinstein, Eric R Meissner, Lars Holdijk, Lukas Schott, Carl-Johann Simon-Gabriel, Apostolos Modas, William Fleshman, Xuefei Ning, [and many others](https://github.com/bethgelab/foolbox/graphs/contributors).

# References

