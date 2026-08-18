# Machine learning operations (MLOps)

So far, much of the Machine Learning (ML) field has focused on the theory; MLOps, or "Machine Learning Operations," focuses on the practice. In this advanced topic module, we bring part of this emerging field as a distilled set of best practices for implementing in your ML projects, promoting solution quality and maintainability. This module covers three phases: Before-Development, Development, and After-Development. In this introductory video, you'll learn why and what is MLOps; furthermore, you'll get an idea of each part that this module offers.

<iframe width="560" height="315" src="https://www.youtube.com/embed/y9mCFlIdr00?si=jdZ32vYXwjwFEms4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Project integration**: Regardless what type of ML algorithm you use, your development process will benefit from including these practices to your workflow. This module will take you from running a couple of ML experiments on your own to manage 100 ML experiments with your team.

## Recorded Lectures

- {doc}`Before_development`
- {doc}`Development` 
- {doc}`After_development`

## Exercises

- {doc}`exercises-clean/EX1-Before_Development`
- {doc}`exercises-clean/EX2-Development`
- {doc}`exercises-clean/EX3-After_Development`

### Format and schedule

This is a self-study topic, with video recordings and Jupyter Notebooks. As such, there is no pre-defined schedule.

## Installing additional libraries

In the following notebooks, you will use two libraries that is are not present in your environment. The first one, ucimlrepo, is a library that allows you to download datasets from the UCI Machine Learning Repository. The second one, wandb, is a library that allows you to track your experiments.
To install them, go in your dsaie conda environment by running in the Anaconda prompt:

    conda activate dsaie

Then, you can install the libraries with these command lines:

    pip install ucimlrepo
    pip install wandb==0.16.1

In case there are compatibility issues ask to a TA for help.

## References and further resources

- [Full Stack Deep Learning - 2022 Course](https://fullstackdeeplearning.com/course/2022/)

- Kreuzberger, D., Kuhl, N., & Hirschl, S. (2023). Machine Learning Operations (MLOps): Overview, Definition, and Architecture. IEEE Access, 11(February), 31866–31879. <https://doi.org/10.1109/ACCESS.2023.3262138>

- [Engineering best practices for Machine Learning](https://se-ml.github.io/practices/)

- Lones, M. A. (2021). How to avoid machine learning pitfalls: a guide for academic researchers. 1–25.<http://arxiv.org/abs/2108.02497>

- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) by Andrej Karpathy

- [Scientific paper using the same dataset as in the exercises](https://www.sciencedirect.com/science/article/pii/S0925400507007691)

- Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., & Young, M. (2014). Machine Learning : The High-Interest Credit Card of Technical Debt. NIPS 2014 Workshop on Software Engineering for Machine Learning (SE4ML), 1–9. <https://research.google/pubs/machine-learning-the-high-interest-credit-card-of-technical-debt/>

- [RainGuru](https://rainguru.tudelft.nl/)
