# Long Short-Term Memory and Gated Recurrent Units

## Brief Overview

In our previous lecture, we looked into the fundamentals of Recurrent Neural Networks (RNNs) and their suitability for handling sequential data. Today's lecture builds upon this foundation by addressing a crucial limitation of simpler RNNs: <u>their struggle with long-term dependencies</u>. The inherent structure of basic RNNs makes it challenging for them to preserve information over extended sequences, leading to difficulties in learning dependencies over long distances. This is mainly due to the gradient diminishing rapidly over time, a phenomenon known as the **vanishing gradient problem**.

To overcome these limitations, we introduce *gated recurrent neural networks*, namely **Long Short-Term Memory (LSTM)** networks and **Gated Recurrent Units (GRU)**. These architectures mitigate the problem of long-term dependencies. They achieve this through the integration of gating mechanisms - a series of **learnable gates** that regulate the flow of information. These gates, namely the input, forget, and output gates in the case of LSTM (and a simplified version in GRU), enable the network to selectively remember and forget information. This selective retention and omission of information is pivotal in maintaining relevant context over long sequences.

One of the significant advantages of these gated mechanisms is their facilitation of gradient flow during training. Unlike simpler RNNs, LSTMs and GRUs maintain a more stable gradient flow. This stability is crucial for effective learning in deep learning models and allows for more robust training over longer sequences.

Like RNNs, LSTMs and GRUs can be adapted into various architectures to suit different requirements. These adaptations include encoder-decoder frameworks, which are particularly useful for sequence-to-sequence tasks, bidirectional configurations that allow the network to access both past and future context in a sequence, and deeper network structures that can capture more complex patterns by processing information at multiple levels.

## Resources

[Slides](https://surfdrive.surf.nl/files/index.php/s/LxaAIofEslOas82)

Chapter 10.7 and 10.10 {bdg-muted}`goodfellow-dl` but NOT the part in <span style="color:red">**RED**</span>

## Backpropagation Through Time (BPTT)
RNNs are trained with **Backpropagation Through Time (BPTT)**, adapting weights based on previous outputs and states. BPTT is performed on the **unfolded RNN**, where sequential layers are treated like traditional ones. BPTT can cause gradients to **vanish** or **explode**, hindering RNN learning.

The diagram below shows the unfolding of the RNN hidden states over time.

```{figure} ../figures/output_per_time_step.svg
:alt: bptt_rnn
:height: 350px
:align: center

Unfolding of RNN with Single Output to Demonstrate Backpropagation
```

Recall the formulas used to compute the hidden state $h^{(t)}$ in a simple RNN and its output $o^{(t)}$:

$$
\mathbf{h}^{(t)} = f(\mathbf{U}^T\mathbf{x}^{(t)} + \mathbf{W}^T\mathbf{h}^{(t-1)} + \mathbf{b})
$$

$$
\mathbf{o}^{(t)} = g(\mathbf{V}^T\mathbf{h}^{(t)} + \mathbf{b}_o)
$$

Let's consider the case of an RNN pattern providing a single output at the end of a sequence. The output at the final time step $t=\tau$ is given by $o^{(\tau)}$, which is computed by looping through the $(\mathbf{x}^{(t)},\mathbf{h}^{(t)})$ tuples one time step at a time. The loss $L^{(\tau)}$ is obtained as a function of the desired output $y^{\tau}$ and the prediction $o^{\tau}$. To adjust the parameters of the RNN, we use backpropagation through time, which requires computing the derivatives of the loss with respect the the mathematical graph of the RNN.

If we consider that $f$ and $g$ are linear, for simplicity, the gradient of the loss with respect to the weights $\mathbf{W}$ is given by:

$$
\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L(y^{(\tau)},o^{(\tau)})}{\partial \mathbf{W}} = 
    \frac{\partial L(y^{(\tau)},o^{(\tau)})}{\partial o^{(\tau)}} \frac{\partial o^{(\tau)}}{\partial \mathbf{h}^{(\tau)}} \frac{\partial \mathbf{h}^{(\tau)}}{\partial \mathbf{W}}
$$

Due to the recurrency, $\mathbf{h}^{(\tau)}$ is a function of both $\mathbf{h}^{(\tau - 1)}$ and $\mathbf{W}$, where the computation of $\mathbf{h}^{(\tau - 1)}$ also depends on $\mathbf{W}$. This also holds for all previous $\mathbf{h}^{(\tau - k)}$, going backwards until $\mathbf{h}^{(1)}$. Therefore, the rightmost term of the equation can be written as follows

<!-- $$
\frac{\partial \mathbf{h}^{(\tau)}}{\partial \mathbf{W}} = \sum_{t=1}^{\tau} \left( \frac{\partial \mathbf{h}^{(\tau)}}{\partial \mathbf{h}^{(t)}} \frac{\partial \mathbf{h}^{(t)}}{\partial \mathbf{W}} \right)
$$

where the dependency on $\mathbf{W}$ at each timestep $t$ arises from the recursive structure of the RNN. Expanding this further using the chain rule, we get: -->

$$
\frac{\partial \mathbf{h}^{(\tau)}}{\partial \mathbf{W}} = \sum_{t=1}^{\tau} \left( \prod_{j=t+1}^{\tau} \frac{\partial \mathbf{h}^{(j)}}{\partial \mathbf{h}^{(j-1)}} \right) \frac{\partial \mathbf{h}^{(t)}}{\partial \mathbf{W}}
$$

The product term reflects the recursive application of the chain rule, accounting for the propagation of gradients backward through all previous time steps. Since the hidden state at each time step depends on the previous one, the gradient must accumulate over all previous hidden states.

The gradient of the loss with respect to $\mathbf{W}$ can therefore be rewritten as:

$$
\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L(y^{(\tau)},o^{(\tau)})}{\partial o^{(\tau)}} \frac{\partial o^{(\tau)}}{\partial \mathbf{h}^{(\tau)}} \sum_{t=1}^{\tau} \left( \prod_{j=t+1}^{\tau} \frac{\partial \mathbf{h}^{(j)}}{\partial \mathbf{h}^{(j-1)}} \right) \frac{\partial \mathbf{h}^{(t)}}{\partial \mathbf{W}}
$$

This formulation emphasizes how gradients flow through the entire sequence, where the chain rule progressively accumulates the dependencies between hidden states and the weights over multiple time steps. This process is the core of backpropagation through time (BPTT) in recurrent neural networks.

To better understand this process, consider an example where for simplicity we consider a single weight $\omega$:

$$
h^{(t)} = f(W \cdot h^{(t-1)} + U \cdot x^{(t)} + b)
$$

$$
= W \cdot h^{(t-1)} + U \cdot x^{(t)} + b
$$

$$
\frac{\partial h^{(t)}}{\partial h^{(t-1)}} = W \approx \omega
$$

Now we can see that the even with relatively short sequences, the gradients can explode or vanish:

$$
\lim_{t \rightarrow \infty} w^{t} = \infty \text{ if } |\omega| > 1
$$
$$
\lim_{t   \rightarrow \infty} w^{t} = 0 \text{ if } |\omega| < 1
$$

## Gated RNNs

Gated RNNs use **gates** and specialized mechanisms to improve the flow of information over long sequences, addressing the limitations of simple RNNs. These gates selectively retain or forget information, enabling better long-term memory and reducing issues like vanishing or exploding gradients. In particular, gated RNNs such as **Long Short-Term Memory (LSTM)** and **Gated Recurrent Units (GRU)** incorporate key features that stabilize gradient propagation during training.

```{figure} ../figures/gated_rnns.svg
:alt: gated_rnns
:height: 450px
:align: center
Gated RNNs Side by Side (normal, LSTM, GRU)
```

### Long Short-Term Memory (LSTM)
Long Short-Term Memory networks (LSTMs) were introduced in 1997 to address the limitations of traditional recurrent neural networks, particularly the vanishing and exploding gradient problems [^1]. LSTMs mainly achieve this by introducing a new component, the **cell state** ($s$), which allows information to propagate across time steps with minimal modifications. Unlike the hidden state, the cell state serves as a "highway" for information, enabling gradients to flow additively instead of multiplicatively, thereby avoiding their degradation over long sequences. Additionally, LSTMs use a set of **gates**—the forget, input, and output gates—to regulate the flow of information into, within, and out of the cell. Each gate is a linear transformation followed by a *sigmoid* activation that scales the information. The `sigmoid` function outputs values between 0 and 1, where 0 means no information passes through the gate and 1 means all information is retained, while partial values allow for selective filtering.

```{figure} ../figures/lstm_gates.svg
:alt: lstm_gates
:height: 300px
:align: center
LSTM Gates Connected In Sequence
```

[^1]: Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural Comput 9.8 (1997): 1735-1780.

#### Gating Mechanisms in LSTM
All gates receive current input $x^{(t)}$ and previous hidden state $h^{(t-1)}$ as inputs. In general, the number of parameters a gate receives as input is $(N_h \times N_i) + (N_h \times N_h) + N_h$, where $N_h$ is the number of hidden units and $N_i$ is the number of input units. 

For example, Figure {numref}`gating_mechanism_lstms` shows a gate for $N_h = 1$, $N_i = 3$, yielding 5 parameters by considering the bias term:

```{figure} ../figures/gating_mechanism_lstms.png
:name: gating_mechanism_lstms
:height: 300px
:align: center

Gating mechanism example.
```

There are three main gates in an LSTM: the *forget* gate, the *input* gate, and the *output* gate. The table below explains the role of each gate in the LSTM architecture.

> Be aware of the different notations between the figures and the description, e.g., using $x_t$ and $x^{(t)}$ to denote the inputs at time step $t$. 

| Gate | Role |
|:-|:-|
| ![Forget gate](../../../images/forget_gate_lstm.svg) | The **forget gate**  $f^{(t)}$ decides what information should be removed from the cell state. | 
| ![Input gate](../../../images/input_gate_lstm.svg) | The **input gate** $i^{(t)}$ decides which values of the cell state to update, while a **tanh** layer creates a vector of new candidate values to add. |
| ![Cell state update](../../../images/update_gate_lstm.svg) | The cell state $s^{(t)}$ (indicated with $C_t$ in the figure) is **updated** based on the outcomes of the two previous operations via a pointwise sum operation. Note that added values can be negative (they come from **tanh**). |
| ![Output gate](../../../images/output_gate_lstm.svg) | The **output gate** $q^{(t)}$ (indicated with $o_t$ in the figure) decides what parts of the cell state will be shared externally by **determining the hidden state**. The cell state first passes through **tanh** for normalization between [-1, 1]. |

LSTM training tunes gate weights to manage data flow, for learning sequences effectively. Optimized gate weights enable more precise memory control. Gates learn to differentiate temporal scales, allowing LSTMs to capture both short and long-term dependencies. LSTMs have **~4 times the parameters** of RNNs because of gates.

While a simple RNN computes its hidden state as:

$$
\mathbf{h}^{(t)} = f(\mathbf{U}^T \mathbf{x}^{(t)} + \mathbf{W}^T \mathbf{h}^{(t-1)} + \mathbf{b}_h),
$$

the LSTM's hidden state computation is:

$$
\mathbf{h}^{(t)} = \tanh (\mathbf{s}^{(t)}) \odot \mathbf{q}^{(t)}
$$

where the cell state $s^{(t)}$ depends on the forget gate $f^{(t)}$ and input gate $i^{(t)}$, $q^{(t)}$ is the output gate, and $\odot$ represents element-wise multiplication. Note how all the gates have a superscript $^{(t)}$, indicating that their output changes with the sequence as it depends on inputs $x^{(t)}$ and previous hidden state $h^{(t-1)}$.

#### How LSTMs handle long sequences
To understand why LSTMs are more effective at handling long sequences, let's dive deeper into the workings of the cell state and how gradients flow through the network.

Previously, we saw that in traditional RNNs information is propagated by multiplying the hidden state with weight matrices. During backpropagation, these multiplications create issues: if a weight is 0.9, after 100 time steps we get $0.9^{100} \approx 0$ (vanishing gradient). Similarly, if a weight is 1.1, after 100 time steps we get $1.1^{100} \approx 13,780$ (exploding gradient). These multiplicative effects prevent the network from learning long-term dependencies.

LSTMs solve the gradient problems through their cell state, which uses addition instead of multiplication to propagate information. Contrary to the *hidden state*, the LSTM's *cell state* update follows a more stable pattern controlled by its gates:

$\mathbf{s}^{(t)} = \mathbf{f}^{(t)} \odot \mathbf{s}^{(t-1)} + \mathbf{i}^{(t)} \odot \tilde{\mathbf{s}}^{(t)}$

where:
- $\mathbf{f}^{(t)}$ is the forget gate (controls what to keep from previous state)
- $\mathbf{i}^{(t)}$ is the input gate (controls what new information to add)
- $\tilde{\mathbf{s}}^{(t)}$ is the candidate state (new information to potentially add)
- $\odot$ represents element-wise multiplication

This additive update mechanism acts like a highway running through the entire sequence: information can enter or exit through carefully controlled on-ramps and off-ramps (the gates), but the highway itself maintains a clear path forward, preserving information integrity over long distances. During backpropagation, this additive nature becomes crucial for learning. When computing gradients to adjust the network's parameters, the error signals flow backward through these additions without suffering from the repeated multiplication that plagues traditional RNNs. Again the "highway" is a suitable metaphor: important information and gradients can travel long distances through the sequence without degrading. The network can then effectively adjust its gate parameters—the weights and biases that control information flow—based on these well-preserved gradient signals, ultimately leading to better optimization of the loss function.


#### Gradient Flow Through the Cell State
To understand why gradients propagate more effectively through additions in LSTMs, let's analyze the gradient flow backward through time. Consider a loss $L$ at time step $T$. We want to compute how this loss depends on the cell state at an earlier time step $t$ using the chain rule:

$$
\frac{\partial L}{\partial \mathbf{s}^{(t)}} = \frac{\partial L}{\partial \mathbf{s}^{(T)}} \prod_{k=t+1}^{T} \frac{\partial \mathbf{s}^{(k)}}{\partial \mathbf{s}^{(k-1)}}
$$

For any time step $k$, using the cell state update rule:

$$
\mathbf{s}^{(k)} = \mathbf{f}^{(k)} \odot \mathbf{s}^{(k-1)} + \mathbf{i}^{(k)} \odot \tilde{\mathbf{s}}^{(k)}
$$

The derivative of the cell state with respect to the previous state simplifies to:

$$
\frac{\partial \mathbf{s}^{(k)}}{\partial \mathbf{s}^{(k-1)}} = \mathbf{f}^{(k)}
$$

Therefore, the gradient from time step $T$ back to $t$ becomes:

$$
\frac{\partial L}{\partial \mathbf{s}^{(t)}} = \frac{\partial L}{\partial \mathbf{s}^{(T)}} \prod_{k=t+1}^T \mathbf{f}^{(k)}
$$

Since the forget gate $\mathbf{f}^{(k)}$ is constrained between 0 and 1 (due to the sigmoid activation), the network can learn to keep it close to 1 when important information needs to be preserved. This additive structure helps prevent the gradients from exploding or vanishing as rapidly as in traditional RNNs. While RNNs can theoretically learn weights to maintain stable gradients, this is usually very difficult in practice. LSTMs are explicitly designed with gating mechanisms to do so more reliably.

### Gated Recurrent Units (GRU)
Gated Recurrent Units (GRUs) were introduced by Cho et al. in 2014 as a simplified alternative to Long Short-Term Memory (LSTM) networks[^2]. While both architectures use gating mechanisms to control information flow, GRUs simplify the LSTM architecture in three key ways:

1. **State Merging**: The GRU merges the LSTM's cell state and hidden state into a single state vector
2. **Gate Reduction**: The GRU combines the LSTM's forget and input gates into a single **update gate** ($\mathbf{z}^{(t)}$)
3. **Reset Mechanism**: The GRU introduces a new **reset gate** ($\mathbf{r}^{(t)}$), replacing the LSTM's output gate, to control how much of the previous state should be used in computing the new candidate state

This simpler architecture, illustrated in Figure {numref}`gru`, reduces the number of parameters compared to LSTMs while still effectively handling long-term dependencies in many tasks. The update gate controls the mix of previous and new state information, while the reset gate is particularly useful when the model needs to forget past information to capture short-term dependencies.

```{figure} ../figures/lstm_vs_gru.svg
:name: gru
:height: 300px
:align: center
GRU vs LSTM
```

[^2]: Cho, Kyunghyun. "On the properties of neural machine translation: Encoder-decoder approaches." arXiv preprint arXiv:1409.1259 (2014).

### Parameter Complexity of Gated RNNs
LSTMs and GRUs both have more trainable parameters than standard RNNs due to the gating mechanisms. Each gate requires its own set of weights for both the input and hidden state connections.

For an LSTM with $N_h$ hidden units and $N_i$ input features:
- Forget gate: $(N_h \times N_i) + (N_h \times N_h) + N_h$
- Input gate: $(N_h \times N_i) + (N_h \times N_h) + N_h$
- Output gate: $(N_h \times N_i) + (N_h \times N_h) + N_h$
- Cell candidate: $(N_h \times N_i) + (N_h \times N_h) + N_h$

Total parameters: $4 \times [(N_h \times N_i) + (N_h \times N_h) + N_h]$

For a GRU with the same dimensions:
- Update gate: $(N_h \times N_i) + (N_h \times N_h) + N_h$
- Reset gate: $(N_h \times N_i) + (N_h \times N_h) + N_h$
- Candidate state: $(N_h \times N_i) + (N_h \times N_h) + N_h$

Total parameters: $3 \times [(N_h \times N_i) + (N_h \times N_h) + N_h]$

This reduction of parameters makes GRUs computationally lighter than LSTMs, often making them a preferred choice for smaller datasets or less complex sequence tasks.

## Summary

This chapter explored the limitations of standard Recurrent Neural Networks (RNNs), primarily their struggle with long-term dependencies due to vanishing and exploding gradients. We introduced the concepts of Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRU) as solutions to this issue. LSTMs address the gradient problem through a cell state and gated mechanisms that allow better gradient preservation. GRUs simplify the architecture while retaining effective control over information flow through update and reset gates.

Key topics covered include the structure of basic RNNs and the gradient flow issues in Backpropagation Through Time (BPTT). The chapter also discussed how LSTMs use gates (forget, input, and output) and a cell state to stabilize gradients, while GRUs reduce complexity with a combined update gate and fewer parameters. Additionally, the mathematical formulation of gradient flow and parameter complexity for both architectures was presented.

```{admonition} Exam questions    
:class: danger    
You are not required to memorize the mathematical details of BPTT or derive its equations during the exam. However, you are expected to understand the core limitations of standard RNNs, such as vanishing and exploding gradients, and explain how LSTM and GRU architectures overcome these limitations through gating mechanisms and cell state management. Focus on conceptual understanding rather than detailed mathematical derivations.
+++    
{bdg-primary}`written-exam`    
```

## References