# Probabilistic Deep Learning

This section focuses on the **dropout** regularization technique and its usage in **Probabilistic Deep Learning**. Dropout, used during the *training phase*, randomly deactivates neurons to prevent over-reliance on specific neurons and encourage a more generalized learning. Building on this technique, **Monte Carlo Dropout**, where dropout is kept active during inference, allows the network to generate multiple forecasts by emulating ensembling. The effectiveness of dropout hinges on tuning the *dropout rate* and selecting the appropriate layers for its application. These decisions impact the balance between regularization and the generation of informative probabilistic forecasts. We will also discuss how to evaluate these probabilistic forecasts using metrics like the *Prediction Interval Coverage Probability (PICP)* and thee *Prediction Interval Normalized Average Width (PINAW)*, focusing on achieving prediction intervals that are accurate in coverage and narrow in width.

## Lectures
- {doc}`dropout`
- {doc}`mcd`

## Exercises
- {doc}`exercises-clean/waterdemand_rnn_part_4`
  
## Quiz
- {doc}`review`
