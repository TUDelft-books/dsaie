# Review and Quiz

## Multiple choice questions
<iframe src="https://tudelft.h5p.com/content/1292036115263376147/embed" aria-label="Generative ML" width="1088" height="637" frameborder="0" allowfullscreen="allowfullscreen" allow="autoplay *; geolocation *; microphone *; camera *; midi *; encrypted-media *"></iframe><script src="https://tudelft.h5p.com/js/h5p-resizer.js" charset="UTF-8"></script>

## In the exam you will also be tested on open questions. Here are a few example questions:

`````{tab-set}
````{tab-item} Open question 1: 
For which special encoder and decoder architecture do the VAE and PPCA amount ot the same statistical model?

Example answer:
```{toggle} Answer:
VAE and PPCA are the same model if the activation functions within both encoder and decoder are linear. The depth of the networks does not matter, as a sequence of linear models is still a linear model.
```
````

````{tab-item} Open question 2: 
You trained your variational autoencoder.
You inspect the results, and see that most of the generations look highly distorted.
You plot your dataset in the latent space, and see clear cluster formations with a lot of unoccupied space inbetween.
What can you do improve model predictions?

Example answer:
```{toggle} Answer:
Your latent space most likely is not regular enough. You can put more emphasis by increasing the parameter `kld_weight`, that indirectly governs the degree of regularization.
```
````
`````