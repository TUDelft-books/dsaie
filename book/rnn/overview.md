# Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are a type of artificial neural network designed to process sequential data, such as time series or natural language. They are particularly useful for problems involving sequential data, as they can remember past information to make predictions about future data points. Key features of RNNs include:

- **Hidden state**: RNNs have a hidden state, which remembers some information about a sequence and serves as a form of memory. This allows the network to capture temporal dependencies in the data.

- **Shared parameters**: Unlike multi-layer perceptrons, RNNs use the same parameters for each input, reducing the complexity of parameters and allowing the model to generalize to sequences of varying lengths.

- **Architecture**: RNNs have a unique architecture that processes input data in a sequential manner, with the output from the previous step being fed as input to the current step. This allows the network to capture temporal dependencies in the data.


RNNs have various applications, such as natural language processing, translation and speech recognition. They have been now replaced in most of these tasks by the more powerful transformer architecture (i.e., GPT). That said, they are still very useful from an educational perspective and they can still be employed succesfully in our field when the amount of data we have is insufficient to exploit the scaling capabilities of transformers. Typical applications include signal processing, time series prediction and forecasting.

For this section, you can refer to Chapter 10 of {bdg-muted}`goodfellow-dl`. [Here](https://surfdrive.surf.nl/files/index.php/s/SjyZ8qZk121D9Jx) you can find a version with highlighted in <span style="color:red">**RED**</span> the parts you DO NOT need.

## Lectures

- {doc}`rnn`
- {doc}`lstm_gru`

## Exercises

- {doc}`exercises-clean/waterdemand_rnn_part_1`
- {doc}`exercises-clean/waterdemand_rnn_part_2`
- {doc}`exercises-clean/waterdemand_rnn_part_3`

## Quiz
- {doc}`review`
