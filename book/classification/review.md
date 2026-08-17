# Review and Quiz

## Multiple choice questions:
<iframe src="https://tudelft.h5p.com/content/1292036112732624177/embed" aria-label="Classification" width="1088" height="637" frameborder="0" allowfullscreen="allowfullscreen" allow="autoplay *; geolocation *; microphone *; camera *; midi *; encrypted-media *"></iframe><script src="https://tudelft.h5p.com/js/h5p-resizer.js" charset="UTF-8"></script>

## In the exam you will also be tested on open questions. Here is an example question:

`````{tab-set}
````{tab-item} Open question 1: 
Explain the main difference between models dealing with 2 classes, and those dealing with >2 classes.

Example answer:exam
```{toggle} Answer:
In a 2-class problem, models can have a single output which can be rounded to for example 0 or 1 to determine the class.
A model classifying >2 classes generally has an output for each class (which is then transformed with a softmax function).
One reason for not having a single output with labels [0, 1, 2, ..] is that it assumes that the class with label 1 is always between class 0 and 2.
```
````
`````