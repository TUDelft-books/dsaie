# Recurrent Neural Networks

## Brief Overview

In today's lecture, we introduce **Recurrent Neural Networks (RNNs)** and their role in processing sequential data. We compare RNNs with Multi-Layer Perceptrons (MLPs), noting a key difference: RNNs use shared weights across time steps, which makes them more efficient for tasks involving time-dependent data. This concept should be familiar to you after you have been working with CNNs last week.

We looked at different ways RNNs are used. Some RNNs produce outputs at every time step, which is useful for tasks that require predicting an output for each element in the input sequence. Others are designed to take a whole sequence of inputs and produce just one output, which is helpful for tasks like summarizing information or making a single prediction at the end of a sequence (e.g., time series analysis).

Then we will discover more advanced RNNs, such as the **Encoder-Decoder** architecture which is particularly suitable for tasks that involve turning one sequence into another, potentially of a different length. We will see that the *encoder* part of the network condenses the input sequence into a latent representation known as context, while the *decoder* uses this summary to generate the target sequence step by step.

We will also briefly introduce **Bidirectional RNNs (Bi-RNNs)**, which read sequences forwards and backwards to better understand the context. However, we will see that they are not suitable for real-time tasks because they need to see all the data, including data from the future, to work. Finally, we will look into **Deep RNNs**, which have multiple layers vertically stacked. These networks may be better at recognizing complex patterns because they can process information at different levels by leveraging an *hierarchical inductive bias*. 

## Resources

[Slides](https://surfdrive.surf.nl/files/index.php/s/EuECF68nZakeXDo)

Chapter 10 until 10.5 {bdg-muted}`goodfellow-dl` but NOT the part in <span style="color:red">**RED**</span>

## What is an RRN?
Recurrent neural networks are a type of artificial neural network designed to recognize patterns in sequences of data, such as text, sensor readings, and other times series data. They are called "recurrent" because because of a hidden array representing "memory" that is passed from one step of the network to the next. This allows RNNs to learn from the past and make predictions about the future.

RNNs learn shared parameters to implement a **sequential inductive bias**:
1. Nearby elements in a sequence are **highly** correlated.
2. Patterns repeat over "time".

The figure below illustrates the progression of the RNN's training process, where the model is being trained to predict streamflow based on meteorological inputs and river catchment characteristics. As training advances, the RNN increasingly captures the periodic patterns observed in the hydrograph[^1].

```{figure} ../figures/images/copyrighted/example_rnn_data.png
:alt: example_rnn_data
:height: 350px
:align: center

Kratzert, Frederik, et al. "Rainfall–runoff modelling using long short-term memory (LSTM) networks." Hydrology and Earth System Sciences 22.11 (2018): 6005-6022
```
[^1]: Hydrographs represent river discharge or flow rate over time.

## RNN vs MLP

Consider the following example in natural language processing that highlights why sequential processing matters. 

Let's say we train both an MLP and an RNN model on sentences containing years, with the task of identifying which number in a sentence represents a year. For example, in the sentence "In **2009** I went to Nepal", both models learn to identify "2009" as a year.

Now, let's test the models on a similar sentence with different word order: "I went to Nepal in **2009**". An MLP may struggle here because of its fundamental architecture: it processes each word position as a separate input feature, without any mechanism to maintain relationships between words in different positions. So when "2009" appears in a different position than what was seen during training, the MLP may not recognize that it serves the same function. Additionally, MLPs require fixed-size inputs, meaning they cannot naturally handle sequences of varying lengths.

In contrast, an RNN is designed to process inputs sequentially, maintaining a hidden state that captures the relationships between words regardless of their position. This hidden state is updated at each time step, allowing the RNN to understand that "2009" is a year whether it appears at the beginning or end of the sequence. Furthermore, RNNs can handle variable-length sequences naturally, as they process one element at a time and can continue this process for any sequence length.

> **Note:** This example is simplified for illustration. In practice, words are not directly fed to neural networks as raw text. Instead, they are first converted into numerical representations: typically, words are mapped to indices in a vocabulary, then transformed into dense vector representations called word embeddings. These embeddings capture semantic relationships between words and serve as the actual input to the neural networks.

**MLPs have the following properties:**
- No loops or memory mechanism
- Process inputs independently and in parallel
- Outputs are computed using:

$$
\mathbf{h} = f(\mathbf{U}^T \mathbf{x} + \mathbf{b}_h)
$$

$$
\mathbf{o} = g(\mathbf{V}^T \mathbf{h} + \mathbf{b}_o)
$$

**RNNs, on the other hand, are characterized by:**
- A loop within the hidden layer
- A memory vector that persists across time steps
- Sequential processing of inputs to account for temporal dependencies 
- Sequential output computation with a time variable $t$ tracking position

$$
\mathbf{h}^{(t)} = f(\mathbf{U}^T \mathbf{x}^{(t)} + \mathbf{W}^T \mathbf{h}^{(t-1)} + \mathbf{b}_h)
$$

$$
\mathbf{o}^{(t)} = g(\mathbf{V}^T \mathbf{h}^{(t)}+ \mathbf{b}_o)
$$

A visual comparison is provided in Figure {numref}`mlp_vs_rnn` below:

```{figure} ../figures/MLP_vs_RNN_structure.svg
:name: mlp_vs_rnn
:height: 350px
:align: center
MLP and RNN structures
```

## Shared weights

Let's now explore a crucial feature of RNNs: shared weights across time steps. 

In RNNs, the same weights are used at each time step to process new inputs (input-hidden weights $U$), update the hidden state (hidden-hidden weights $W$), and generate outputs (hidden-output weights $V$). This weight sharing is fundamental because it allows the network to detect the same patterns regardless of where they occur in the sequence. It enables the model to generalize to sequences of any length and encodes the assumption that the same rules should apply throughout the sequence. For instance, in language, the relationship between words follows the same grammar rules regardless of their position.


### Number of parameters

Let's compare this with MLPs and see how it affects the number of parameters. Consider a scenario with input sequences of length $T$ (time steps), $N_i$ features at each time step, $N_h$ hidden units, and $N_o$ output units.

For an MLP to process a sequence:
- It needs separate weights for each time step, leading to input-hidden weights $U: {N_h \times (N_i \times T)}$
- It has hidden-output weights $V: {N_o \times N_h}$
- It has biases $b_h: {N_h}$ and $b_o: {N_o}$

An RNN, through weight sharing across time steps, has:
- Input-hidden weights $U: {N_h \times N_i}$ (shared across all time steps)
- Hidden-hidden weights $W: {N_h \times N_h}$ (enabling memory across time steps)
- Hidden-output weights $V: {N_o \times N_h}$
- Biases $b_h: {N_h}$ and $b_o: {N_o}$

Let's see a concrete example with sequence length $T = 10$, features per time step $N_i = 20$, hidden units $N_h = 30$, and output units $N_o = 2$.

MLP parameters:
- Input-hidden weights: $30 \times (20 \times 10) = 6,000$ parameters
- Hidden-output weights: $2 \times 30 = 60$ parameters
- Biases: $30 + 2 = 32$ parameters
Total: 6,092 parameters

RNN parameters:
- Input-hidden weights: $30 \times 20 = 600$ parameters
- Hidden-hidden weights: $30 \times 30 = 900$ parameters
- Hidden-output weights: $2 \times 30 = 60$ parameters
- Biases: $30 + 2 = 32$ parameters
Total: 1,592 parameters

Beyond the reduction in parameters (1,592 vs 6,092), the shared weights in RNNs enable the network to learn patterns that can be recognized at any position in the sequence, process sequences of arbitrary length during inference, and maintain consistency in how temporal patterns are processed. This example illustrates how the number of parameters in the MLP grows linearly with the sequence length T, while the RNN maintains a constant number of parameters regardless of sequence length. Linked to the **curse of dimensionality**, this linear scaling in MLPs can still lead to practical issues: longer sequences require more parameters, which in turn require more training data and computational resources, and can lead to overfitting.

## 1D Convolutions vs RNN
1D convolutional layers also implement a **sequential inductive bias** and share parameters. However, they have a **shallow** structure where the focus is on nearby elements only. In contrast, RNNs have a **deep** structure where the focus is on the entire sequence. 

1D convolutions produce an output for each layer that is a function of a small number of **neighbouring** inputs. This is useful for tasks that require capturing local patterns in the data, such as detecting edges in images. RNN layers share parameters differently:
* Each output is derived from previous outputs using the same update rule.
* **Memory** creates a deep computational graph. 

The differences between 1D convolutions and RNNs can be seen clearly with the following diagram for the computation of a 1D convolutional layer where it takes a function of nearby inputs, but not the entire sequence as an RNN would sequentially. 

```{figure} ../figures/1d_conv.svg
:alt: 1d_conv
:height: 350px
:align: center

1D Convolutional Layer
```

## RNN Hidden Layer as a Dynamical System
The RNN hidden layer updates iteratively, reflecting a state-dependent, time-evolving process. The hidden layer can be thought of as a **dynamical system** that evolves over time, represented by the following equation:

$$  
\mathbf{h}^{(t)} = f(\mathbf{h}^{(t-1)}, \mathbf{x}^{(t)}; \mathbf{\theta})  
$$

where $\mathbf{h}^{(t)}$ is the hidden state at time $t$, $f$ is the update function, $\mathbf{x}^{(t)}$ is the forcing external input at time $t$, and $\theta$ are the parameters of the network,

$$ 
\mathbf{\theta} = \{\mathbf{U}, \mathbf{W}, \mathbf{b}_h\}  
$$

Using the same update function $f$ with the same parameters $\theta$, the hidden state is computed with $f$ receiving the same input size at each time step. The hidden layer can be "unfolded" to show how the current hidden state depends on all previous states and inputs:

$$
\mathbf{h}^{(t)} = f(\mathbf{h}^{(t-1)}, \mathbf{x}^{(t)} ; \mathbf{\theta})
$$

$$  
= f(f(\mathbf{h}^{(t-2)}, \mathbf{x}^{(t-1)} ; \mathbf{\theta}), \mathbf{x}^{(t)} ; \mathbf{\theta}) 
$$

$$   
= f(f(f(\mathbf{h}^{(t-3)}, \mathbf{x}^{(t-2)} ; \mathbf{\theta}), \mathbf{x}^{(t-1)} ; \mathbf{\theta}), \mathbf{x}^{(t)} ; \mathbf{\theta})  
$$

This unfolding reveals how information from previous time steps flows through the network, with each state depending on all previous states and inputs.

## RNN Design Patterns
RNNs can be designed to handle different types of sequence-based tasks. The number of outputs can be independent of the sequence length. Here are two common design patterns:

### RNN producing outputs at each time step
This design pattern processes a sequence and generates outputs as it processes each element, as illustrated in figure {numref}`output_per_time_step`. For each time step $t$, the RNN produces an output vector $\mathbf{o}^{(t)}$ that is compared against a target vector $\mathbf{y}^{(t)}$ through a loss function $L$. Note that the dimension of each output vector can be different from the dimension of the input sequence elements.

```{figure} ../figures/output_per_time_step.svg
:name: output_per_time_step
:height: 350px
:align: center
RNN producing outputs at each time step.
```

### RNN producing outputs after processing the sequence
This design pattern processes an entire sequence before producing outputs, as illustrated in figure {numref}`single_output_rnn`. The loss function $L$ evaluates the final output vectors $\mathbf{o}^{(T)}$ against target vectors $\mathbf{y}^{(T)}$. This architecture is particularly useful when the task requires understanding the entire sequence before making predictions.

```{figure} ../figures/single_output_rnn.svg
:name: single_output_rnn
:height: 350px
:align: center
RNN producing output after processing the entire sequence.
```

The key difference between these patterns lies in when outputs are generated and used: the first pattern provides outputs as it processes each element, while the second pattern processes the entire sequence before generating outputs. In both cases, the dimension and number of outputs are determined by the task requirements, not by the input sequence length.

It's worth noting that the first pattern (outputs at each time step) is less common in practice as a standalone architecture. This is because many sequence processing tasks involve varying input and output sequence lengths, that are handled with more complex architectures like encoder-decoder models (discussed later). That said, this pattern allows building deeper RNNs, where the output at each step serves as input for deeper layers. 

The second pattern (output after processing the sequence) has found widespread practical applications, especially for predictive tasks. A notable example is its marked impact on lumped rainfall-runoff modeling and riverine flood forecasting (see for instance the works of F. Kratzert and other authors, starting from [^2]). RNNs can process an entire year of meteorological data (365 daily measurements) along with static catchment characteristics, capturing the complex temporal dependencies needed to predict daily runoff and water levels in rivers.

[^2]: Kratzert, Frederik, et al. "Rainfall–runoff modelling using long short-term memory (LSTM) networks." Hydrology and Earth System Sciences 22.11 (2018): 6005-6022.

### Advanced RNN Design Patterns
We can build more complex RNNs, some of which are very important for specific applications due to their ability to handle more complex tasks.

#### Encoder-Decoder RNN
* Map input sequence of length $T_1$ to output sequence of length $T_2$, making this a **sequence-to-sequence** learning task.
* **Encoder:** Processes input sequences to form a context-rich fixed-length vector representation (i.e., latent representation, capturing sequence essence).
* **Decoder:** Generates a target sequence from encoded vector.
* Highly effective for tasks requiring complex mappings between variable-length input and output sequences, similar to what we saw with CNNs such as U-NET.

Encoder-Decoder networks are the basis of machine translation between languages, as a word-by-word approach fails to capture a contextual understanding and syntactic coherence, such as maintaining logical and gramatical sentence structure over many words. 

The final hidden state of the encoder RNN represents a semantic summary of the input sequence, which we can define as the context $C$.
This $C$ is then provided to the decoder RNN as initial state of the decoder RNN and/or as input to the hidden units at each time step.

```{figure} ../figures/encoder_decoder_rnn_diagram.svg
:alt: encoder_decoder_rnn_diagram
:height: 350px
:align: center

Encoder-Decoder RNN
```

A recent breakthrough application of encoder-decoder RNNs has been achieved by researchers at Google in the field of riverine flood forecasting [^3]. Building upon the foundational work in [^2] mentioned before, they developed a state-of-the-art Google FloodHub early warning system. In their system, the encoder processes a full year (365 days) of meteorological variables along with catchment characteristics. The decoder then generates forecasts up to 7 days ahead by unrolling the sequence and incorporating additional inputs from weather forecasting systems.


[^3]: Nearing, Grey, et al. "Global prediction of extreme floods in ungauged watersheds." Nature 627.8004 (2024): 559-563.

#### Bi-Directional RNN
* Processes data in both directions with **two separate hidden layers**, enhancing context awareness.
* Integrates information from both temporal directions, enhancing performance on various sequential data tasks.
* Bi-RNNs require **“future data”**; i.e., this architecture is unsuitable for generation of text or code (like what done by ChatGPT, for instance).
* Used where the entire input sequence is known beforehand.


```{figure} ../figures/bidirectional_rnn.svg
:alt: bi_rnn_diagram
:height: 450px
:align: center

Bi-directional RNN
```

As we see in the figure above, the $h$ recurrence propagates information forward in time (towards the right) while the $g$ recurrence propagates information backward in time (towards the left). The final output is a combination of the two hidden states. 

To better understand the usefulness of processing inputs bidirectionally, consider the following example using machine translation again, where we are provided with an entire sentence and want to predict an adjective between "am" and "tired" based on the sentence. Here, both past and future context is crucial to predict this word, so a Bi-RNN would be useful. 

1. I am **not** tired, I can play another game!
2. I am **very** tired, I think I am going to sleep...

Note how left-to-right processing alone would not get the correct prediction.

#### Deep RNN
* RNN layers **stacked vertically**, maintaining consistent sequence length throughout.
* Successive layers receive prior outputs, forming a **hierarchical representation** of temporal features.
* Hierarchical learning process enhances temporal abstraction.
* Often **computationally intensive**, requiring careful training to avoid issues. As such, rarely RNNs are as deep as CNNs.

```{figure} ../figures/deep_rnn.svg
:alt: deep_rnn
:height: 450px
:align: center

Deep RNN
```

## Summary
Recurrent Neural Networks (RNNs) are specifically designed to process sequential data, leveraging their architecture to capture time-dependent patterns. Their key innovation lies in using shared weights across time steps, making them more efficient than MLPs for sequence modeling. RNNs can be configured in different ways: while they can theoretically produce outputs at each time step, they are most commonly used to map entire sequences to single outputs for tasks like prediction and sequence summarization. For more complex tasks requiring variable-length input and output sequences, the Encoder-Decoder architecture has proven particularly effective. In this setup, an encoder RNN compresses the input sequence into a fixed-length context vector, which a decoder RNN then uses to generate the target sequence. Advanced variants include Bi-directional RNNs, which process sequences in both directions to capture richer context (though unsuitable for real-time applications), and Deep RNNs, which stack multiple layers to handle more complex patterns through hierarchical processing.

## References