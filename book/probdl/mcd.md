# Monte Carlo Dropout

## Resources
[Slides](https://surfdrive.surf.nl/files/index.php/s/amyPZQpMv7URk4Z)

## Brief overview

Monte Carlo Dropout (MCD) extends the standard dropout technique to inference time, enabling uncertainty estimation in deep learning predictions. This approach is particularly valuable for applications requiring reliable predictions. One of the main advantages of MCD is its computational efficiency. Instead of training multiple separate networks to estimate uncertainty, which is computationally expensive, MCD leverages a single network with dropout applied during inference. This makes it feasible to implement uncertainty estimation in large-scale deep learning models without a significant increase in computational resources. Empirical studies have demonstrated the effectiveness of MCD in uncertainty estimation. In the seminal work by Gal & Ghahramani (2016)[^1], the authors demonstrated that MCD approximates a Bayesian approach to DL, which is mathematically grounded, but usually comes with a prohibitive computational cost.


[^1]: Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. PMLR, 2016.

## Understanding Uncertainty in Machine Learning

In machine learning, uncertainty can be broadly categorized into two types:

- **Aleatoric Uncertainty:** This is the inherent noise in the data or measurements. It arises from factors that cannot be reduced by collecting more data, such as sensor noise or inherent variability in the data generation process.

- **Epistemic Uncertainty:** This uncertainty stems from the model's parameters or structure. It reflects our lack of knowledge about the best model to use and can be reduced by gathering more data or improving the model architecture.


## How Monte Carlo Dropout Works

Monte Carlo Dropout specifically targets epistemic uncertainty by approximating Bayesian inference. Instead of disabling dropout during testing (as is done in standard dropout), MCD keeps dropout active and performs multiple forward passes with the same input. Each forward pass randomly drops different neurons, effectively simulating predictions from different network configurations. This process creates an ensemble of predictions, which approximates a distribution over possible models.

The dropout rate is a crucial hyperparameter in MCD:

- **Higher Dropout Rates:** Lead to more diverse predictions by dropping more neurons. This can capture a wider range of model uncertainties but might result in underfitting if too many neurons are dropped.
  
- **Lower Dropout Rates:** Produce more conservative estimates with less diversity in predictions. While this reduces the risk of underfitting, it might not capture the full extent of epistemic uncertainty.

The choice of which layers to apply dropout affects the types of features being regularized:

- **Early Layers:** Impact low-level feature extraction, such as edges and textures in image data.
  
- **Later Layers:** Influence high-level representations, such as object categories or complex patterns.

Selecting the appropriate layers for dropout can help balance the trade-off between model complexity and uncertainty estimation.

## Limitations and Alternatives

### Limitations

- **Partial Uncertainty Capture:** MCD primarily captures epistemic uncertainty and may not fully account for aleatoric uncertainty.
  
- **Hyperparameter Sensitivity:** The choice of dropout rate and the number of forward passes ($k$) can significantly influence the results, requiring careful tuning.

### Alternatives

- **Deep Ensembles:** Training multiple independent models to capture uncertainty. This approach can provide robust uncertainty estimates but is computationally more expensive.
  
- **Bayesian Neural Networks:** Incorporating probability distributions over weights for a more principled Bayesian approach to uncertainty estimation. Very expensive computationally.

## Prediction Intervals (PIs)

Prediction Intervals (PIs) are a way to quantify the uncertainty in our predictions. They provide a range of values within which we expect the true value to lie with a certain probability. 

### Building PIs
To build the prediction intervals, we follow this procedure:

1. **Get the forecasts** for the $k$ models in the ensemble (with MCD, this means testing the model $k$ times with dropout active).
2. **Select the level of confidence** for the PI, typically $95\%$.
3. **Sort the predictions** in ascending order.
4. **Identify the percentiles** to get the lower and upper bounds of the prediction interval. These are the $2.5$th and $97.5$th percentiles for the $95\%$ confidence level.

For example, if we have $100$ predictions, we would look at the 3rd ($2.5\%$ of $100$) and 98th ($97.5\%$ of $100$) predictions in our sorted list.

PIs should meet a **target coverage** and be as **narrow** as possible.

- **Coverage** is the proportion of actual observations falling within the prediction intervals and is a measure of how well they capture the true outcomes. It is linked to *reliability*, as it implies that the model's predictions can be trusted to encompass the true values a high percentage of the time.
  
- **Narrowness** is the width of the prediction interval. Narrower intervals indicate more precise predictions. This is linked to the *informativeness* of the forecast, as more narrow intervals provide more specific, detailed forecasts, which can be more useful in decision-making.

There is a trade-off between coverage and narrowness, as achieving very high coverage can sometimes lead to wider intervals, which might reduce their informativeness and vice versa.

#### Impact of Dropout Rate on Uncertainty Estimates
The dropout rate in Monte Carlo Dropout controls the trade-off between coverage and narrowness of uncertainty estimates:
- **Higher dropout rates** create more diverse predictions, leading to wider prediction intervals with better coverage but potentially less precise estimates
- **Lower dropout rates** produce more consistent predictions, resulting in narrower intervals that are more precise but might miss the true value

The optimal dropout rate should balance these competing objectives: maintaining sufficient coverage of the true value while providing informative (narrow) prediction intervals.

### Evaluating PIs

Several metrics can be used to evaluate the quality of PIs. Two commonly used are the *Prediction Interval Coverage Probability (PICP)* and the *Prediction Interval Normalized Average Width (PINAW)*. The **PICP** s a measure of **coverage**, which measures the proportion of actual observations falling within the prediction intervals. 

$$
PICP = \dfrac{1}{N} \sum^N_{i=1} I \left( y_i \in [L_i, U_i] \right)
$$

where:
* $N$ is the total number of observations
* $Y_i$ is the actual value of the i-th observation
* $L_i$ and $U_i$ are the lower and upper bounds of the prediction interval for the i-th observation
* $I(\cdot)$ is the indicator function that returns 1 if $Y_i$ falls within $[L_i, U_i]$, and $0$ otherwise

The **PINAW** is a measure of **narrowness**, which measures the average width of the prediction intervals normalized by the range of the actual values.

$$
PINAW = \dfrac{1}{R \cdot N} \sum^N_{i=1} \left( U_i - L_i \right)
$$

where all parameters are the same except for $R$, which is the range of the target variable (i.e., the difference between the maximum and minimum observed values).

### From Intervals to Point Predictions
While prediction intervals provide valuable uncertainty information, many applications require single point predictions. There are several methods to derive point predictions from the ensemble of predictions generated, for instance, via MCD:

**1. Mean Prediction**
The most common approach is to use the mean of the ensemble predictions:

$$\hat{y} = \frac{1}{k}\sum_{i=1}^k \hat{y}_i$$

where $k$ is the number of forward passes and $\hat{y}_i$ is the prediction from the i-th forward pass.

**2. Median Prediction**
The median can be more robust to outliers than the mean:

$$\hat{y} = \text{median}(\{\hat{y}_1, ..., \hat{y}_k\})$$

**3. Mode Prediction**
For continuous variables with multiple modes in their distribution:

$$\hat{y} = \text{mode}(\{\hat{y}_1, ..., \hat{y}_k\})$$

The choice between these methods depends on:
- The distribution of predictions (symmetric vs. skewed)
- The presence of outliers in the predictions
- The cost of different types of errors in your application

For example, in structural health monitoring where overestimating a bridge's load-bearing capacity could compromise safety, you might choose the median over the mean if the prediction distribution is right-skewed, as it would be less sensitive to occasional very high predictions that could lead to dangerous overconfidence in the structure's strength.

## References