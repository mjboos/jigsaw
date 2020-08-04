# Classifying Toxic Comments using NLP

This repository contains all code used to generate my solution for the [Jigsaw Toxic Comments Kaggle challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
This was my first Kaggle challenge that I took seriously and I managed to get into the top 3% (119th place out of 4550 teams) - nothing to brag about, but a great learning experience nonetheless.
The goal of this Kaggle challenge was to classify if a given comment was toxic - and in what sense. I.e., it was a [multilabel classification](https://en.wikipedia.org/wiki/Multi-label_classification) that required predicting labels such as toxic, insulting, etc.

## Lessons learned
What stands out to me from this challenge are two things:
- The need for clever ensembling of models, and that I started much too late with that - for most of the competition I used only a single LSTM architecture with attention and ranked quite highly, but as soon as the first ensemble appeared my model was not competetive anymore. Especially in this Kaggle challenge, diversity in models was key.
- Data augmentation. One big boost to model performance - that I unfortunately didn't use - was translating all comments into a different language and then backtranslating it for training. Since we had a relatively large training set, I didn't think this would make a difference, but I was wrong; this technique gave a huge boost in model performance.
- Start with a bomb-proof validation procedure. In the beginning my "workflow" consisted of trying different model architectures by hand and looking for improvements on a hold-out dataset. Only later did I implement a more rigorous cross-validation procedure that could also be used for ensembling. Unsurprisingly, I should've started directly working on this validation procedure, and only then iterated through different model architectures - but doing this model iteration can be actually quite addicting.
