# Physics-Informed Neural Networks

## Brief Overview

Today's lecture focuses on **Physics-Informed Neural Networks (PINNs)**, a type of neural network that <u>incorporates physical laws into its structure</u>. This approach is particularly useful when dealing with systems where the underlying physics, typically described by *Ordinary or Partial Differential Equations* (ODEs/PDEs), are known or sufficiently understood.

PINNs are gaining attention in many areas related to our work. Their unique feature is the integration of physical principles into the learning algorithm, guiding the network towards solutions that are not only consistent with the observed data but also with the known laws of physics. To implement this, the **loss function** of PINNs is more complex than that of conventional neural networks. It typically includes terms that enforce the network's adherence to the physical equations governing the system, in addition to the usual data-driven terms like mean squared error. This ensures that the network's output is both consistent with the data and physically plausible. By embedding physical knowledge, PINNs can be more data-efficient, often requiring fewer data points to train effectively compared to traditional neural networks, as they leverage the underlying physics to supplement the learning process.

One interesting application of PINNs is in parameter **inversion** tasks, where the goal is to determine unknown system parameters or inputs based on observed data. By focusing on the physics-based terms in the loss function, PINNs can potentially reverse-engineer the problem, identifying these unknown parameters.

For practical implementation, PINNs often utilize Multi-Layer Perceptrons (MLPs). This choice is influenced by the fact that unlike domain discretization methods such as grid-based approaches (common in CNNs) or sequence-based techniques (used in RNNs), MLPs treat each input variable as a continuous, mirroring the original formulation of the physical equations they are meant to solve.

To illustrate PINNs, we will work through an example involving a 1D dampened oscillator. This example will show how PINNs can model systems governed by well-defined differential equations. We'll look at how to set up and train a PINN, including designing the loss function and incorporating physical laws into the model.

## Inductive Biases
Previously, we learned about inductive biases. These are assumptions that guide the learning process, helping the model generalize better. For example, we saw **spatial inductive bias** in Convolutional Neural Networks (CNNs) that exploit local correlations in images. We also saw the **hierarchical inductive bias** and **sequential inductive bias** in Recurrent Neural Networks (RNNs). 

In the context of PINNs, we consider **physics as an inductive bias**. This ensures the model adheres to **fundamental laws** of nature (including laws governing chemistry, biology, etc.) By leveraging the properties of physics, we **reduce data requirements**, as we focus learning on physically plausible solutions and patterns. This may yield more **interpretable** and **robust** predictions. Figure {numref}`pinn_diagram` illustrates the comparison between traditional data-driven approaches and physics-informed ones. PINNs are a variant of the latter where both the architecture and the loss function are tailored to the application.

```{figure} ../figures/dd_vs_pinn_diagram.png
:name: pinn_diagram
:height: 300px
:align: center

Data Driven AI Models vs. Physics-Informed AI Models
```

## Basics of PINNs
There are several ways to include physics-related inductive biases in deep learning models, but we will focus on **PINNs (physics-informed neural networks)** which are specializes neural networks that embed physical laws directly into their learning algorithms. PINNs usually learn by minimizing a loss function that includes separate terms for data fit and adherence to governing equations (i.e. ODEs or PDEs). This allows the model to learn from both data and physics, making it more data-efficient and physically consistent.

PINNs have been found to be especially useful in fields where the computational cost of solving PDEs is high, such as:
* Computational Fluid Dynamics
* Structural Mechanics
* Hydraulic Engineering
* Water Management
* Traffic State Estimation
* Seismic Wave Modelling

Some examples PINN applications are highlighted in the figure {numref}`pinn_examples`.


```{figure} ../figures/copyrighted/pinn_examples.png
:name: pinn_examples
:height: 300px
:align: center

PINN examples: (a) Raissi et al. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational physics 378 (2019): 686-707;
(b) Ren, Pu, et al. "SeismicNet: Physics-informed neural networks for seismic wave modeling in semi-infinite domain." Computer Physics Communications (2023): 109010; (c) Ye, Jiawei, et al. "Physics-informed neural networks for hydraulic transient analysis in pipeline systems." Water Research 221 (2022): 118828.
```

## The PINN Framework

The PINN framework (see figure {numref}`pinn_framework`) consists of a neural network that integrates a data-driven approach with physics-informed constraints through boundary/initial conditions, and residuals on ODE/PDE equations. The neural network approximates the solution to the ODE/PDE by taking inputs usually in the form of spatial and temporal coordinates, and producing outputs that represent the values of the ODE/PDE at those coordinates. When labeled training data is available, a data loss is computed between the network’s predictions and the observed values of the target variable(s). Simultaneously, physics-based loss terms, are computed. These are derived from the residuals of the ODE/PDE at selected points in the domain, as well as residuals with respect to boundary conditions and initial conditions. By minimizing these terms during training, PINNs try to predict values that are consistent with the underlying physics. To compute these additional physics-based losses, we  need to **approximate the derivatives** of the output with respect to the inputs. This is done by leveraging the [automatic differentiation capabilities](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) of Deep Learning frameworks such as PyTorch. As usual, backpropagation adjusts the neural network parameters based on all the loss components we specified. The **total loss** which the optimizer uses to update weights is calculated as the weighted sum of the data loss, boundary/ initial condition loss, and residual loss, where the weights are specified as hyperparameters $\omega_{\mathcal{D}}$, $\omega_{\mathcal{B}}$, $\omega_{\mathcal{F}}$. 

```{figure} ../figures/pinn_framework_figure.png
:name: pinn_framework
:height: 300px
:align: center

Overview of the PINN framework
```

### Loss function of PINNs

The **data loss** is computed in the same way as traditional data-driven neural networks, e.g., the MSE between the predicted values and training data. Consider a training dataset of $N_{\text{data}}$ points, with input variable $z_i$, target variable $u_{\text{i}}$, and model prediction $\hat{u}_\theta(z_i)$. The data loss is then:

$$
\text{Data Loss:} \quad \mathcal{L}_{\mathcal{D}} = MSE_{\mathcal{D}} =  \frac{1}{N_{\text{data}}} \sum_{i=1}^{N_{\text{data}}} \left\| \hat{u}_\theta(z_i) - u_{\text{i}} \right\|^2
$$

The **boundary/ initial condition loss** measures how the model respects $N_{\mathcal{B}}$ given boundary and initial conditions, which we can generically denoted by a function $\mathcal{B}$. The boundary/ initial condition loss is then:

$$
\text{Boundary/Initial Condition Loss:} \quad \mathcal{L}_{\mathcal{B}} = \frac{1}{N_{\mathcal{B}}} \sum_{i=1}^{N_{\mathcal{B}}} \mathcal{B}(\hat{u}_\theta(z)) ^2
$$

The **residual loss** is the MSE of the residual of the PDE/ODE equation, which is the difference between the predicted value and the value computed by the PDE/ODE at sampled points which are independent of the data loss. This loss term ensures that the neural network learns the underlying physics of the system. We can consider a function $\mathcal{F}$ which computes this residual based on the differential equation we are using. The residual loss is then:

$$
\text{Residual ODE/PDE Loss:} \quad \mathcal{L}_{\mathcal{F}} = MSE_{\mathcal{F}} = \frac{1}{N_{\mathcal{F}}} \sum_{i=1}^{N_{\mathcal{F}}} \mathcal{F}(\hat{u}_\theta(z)) ^2
$$

Finally, the **total loss** can be computed with a weighted average of loss components.

$$
\text{Total Loss:} \quad \mathcal{L}_{\text{total}} = \omega_{\mathcal{D}} \mathcal{L}_{\mathcal{D}} + \omega_{\mathcal{B}} \mathcal{L}_{\text{B}} + \omega_{\mathcal{F}} \mathcal{L}_{\text{F}}
$$ 

### Types of PINN Tasks and Required Loss Components

The loss components needed for a Physics-Informed Neural Network (PINN) depend on the specific task it is solving. Below is an overview of common PINN tasks and the corresponding loss terms.

#### **1. Simulation (Forward Problems)**
- **Description**: Solve ODEs/PDEs to predict system behavior in the absence of labeled data.
- **Required Loss Components**:
  - **Boundary/Initial Condition Loss** ($\mathcal{L}_{\mathcal{B}}$): Ensures the predicted solution aligns with the required physical constraints at boundaries and initial conditions.
  - **Residual Loss** ($\mathcal{L}_{\mathcal{F}}$): Enforces consistency with the governing equations across the domain.
- **Not Needed**:
  - **Data Loss** ($\mathcal{L}_{\mathcal{D}}$).

#### **2. Inverse Problems**
- **Description**: Estimate unknown parameters or reconstruct solutions using partial or noisy data (see figure {numref}`pinn_framework`).
- **Required Loss Components**:
  - **Data Loss** ($\mathcal{L}_{\mathcal{D}}$): Ensures the model predictions align with the observed data.
  - **Residual Loss** ($\mathcal{L}_{\mathcal{F}}$): Enforces the underlying physics from the governing equations.
- **Optional**:
  - **Boundary/Initial Condition Loss** ($\mathcal{L}_{\mathcal{B}}$): May be included if additional constraints are available.

#### **3. Hybrid Models (Data + Physics)**
- **Description**: Combine labeled data and physics-based constraints for improved modeling in partially observed systems.
- **Required Loss Components**:
  - **Data Loss** ($\mathcal{L}_{\mathcal{D}}$): Fits the model predictions to the labeled data.
  - **Residual Loss** ($\mathcal{L}_{\mathcal{F}}$): Ensures physical consistency based on the governing equations.
- **Optional**:
  - **Boundary/Initial Condition Loss** ($\mathcal{L}_{\mathcal{B}}$): Useful when parts of the domain lack data or require explicit boundary conditions.

### Summary Table

| Task                     | $\mathcal{L}_{\mathcal{D}}$ | $\mathcal{L}_{\mathcal{B}}$ | $\mathcal{L}_{\mathcal{F}}$ |
|--------------------------|-----------------------------|-----------------------------|-----------------------------|
| Simulation               | Not Needed                 | Required                    | Required                    |
| Inverse Problems         | Required                   | Optional                    | Required                    |
| Hybrid Models            | Required                   | Optional                    | Required                    |

## Example: 1D Dampened Harmonic Oscillator
To see how the PINN framework works in practice, consider the 1D dampened harmonic oscillator. 

<center>
  <img src="https://benmoseley.blog/wp-content/uploads/2021/08/oscillator.gif" alt="pinn_diagram" height="150px" />
</center>
<p style="text-align: center;">Animation of 1D Dampened Harmonic Oscillator from https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/</p>

If we were to use purely a data-driven approach, we neural network would not be able to learn the underlying physics of the system, and would fail to accurately forecast future points. If you tried to predict the position of the oscillator at a future time, the model would not be able to capture the oscillatory behavior, as shown below:

<center>
  <img src="https://benmoseley.blog/wp-content/uploads/2021/08/nn.gif" alt="pinn_diagram" height="200px" />
</center>
<p style="text-align: center;">Data-Driven NN predictions from https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/ </p>


This prediction fails to capture the oscillations because the loss function is only focused on the data, 

$$
\dfrac{1}{N} \sum_i^N (u_{\text{predicted}} - u_{\text{actual}})^2
$$

To instead use a PINN to predict the position of the oscillator at a future time, we first need to define the governing ODE and initial conditions. The ODE for a dampened harmonic oscillator is given by:

$$
m \dfrac{d^2x}{dt^2} + \mu \dfrac{dx}{dt} + kx = 0
$$

and the initial conditions are:

$$
x(0) = 1 \quad \text{and} \quad \dfrac{dx}{dt}(0) = 0
$$

where:
* $m$ is the mass of the oscillator
* $\mu$ is the coefficient of friction
* $k$ is the spring constant

Now, the loss function includes the training data (initial points) and the physics, so the total loss function is:

$$
\dfrac{1}{N} \sum_i^N (u_{\text{predicted}} - u_{\text{actual}})^2 + \dfrac{1}{M} \sum_j^M (\left[ m \dfrac{d^2}{dx^2} + \mu \dfrac{d}{dx} + k \right] u_{\text{predicted}})^2
$$

This leads to the following PINN predictions:

<center>
  <img src="https://benmoseley.blog/wp-content/uploads/2021/08/pinn.gif" alt="pinn_diagram" height="200px" />
</center>
<p style="text-align: center;">PINN predictions from https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/  </p>

#### Solving Inverse Problems

Inverse problems are those where we have the output data but need to determine the input parameters. PINNs can be used to solve inverse problems by incorporating the physics of the system into the loss function. This allows the model to learn the underlying physics and infer the unknown parameters from the observed data. In the content of the dampened harmonic oscillator, we could use PINNs to determine the mass, friction coefficient, and spring constant of the oscillator based on the observed data. Below is a diagram illustrating how a PINN can estimate these coefficients to match the observed data.

```{figure} ../figures/inverse_problem_pinn.png
:height: 300px
:align: center

Inverse problem from https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/ 
```

