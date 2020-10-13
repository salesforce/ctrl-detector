CTRL-detector
=====================

This directory contains the code for working with the CTRL output detector model, obtained by fine-tuning a
[RoBERTa model](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/)
with the outputs of the [CTRL model](https://github.com/salesforce/ctrl) and
 [GPT-2 model](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). 
 Please see the
 [model card](https://github.com/salesforce/ctrl-detector/blob/master/ModelCard.pdf) for additional details.

## Downloading a pre-trained detector model

Download the weights for the fine-tuned `roberta-large` detector model (1.5 GB) as shown below. This model performs optimally for
 longer text inputs (> 32 tokens minimum), e.g., a news article.

```bash
wget https://storage.googleapis.com/sfr-ctrl-detector/combine_256.pt
```

## Running a detector model

Please see `example.py` for example code to run the detector model.  
