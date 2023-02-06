# manualist

When it comes to Natural Language Processing, text and oral speech are just a start; we still have the whole manual world to explore! The ideal manual language AI agent would be one fluent in common sign languages and capable of quicly learning others. This is my attempt to build an end-to-end system for American Sign Language to English translation - a modest, yet surprisingly challenging step towards the ideal.

The pipeline for this translation system is as follows:
1. Instead of using the entire image for classification, we utilize MediaPipe to track hand and pose positions, and only use these landmarks. For now, only head position is kept from the pose tracking system. Using separate predictors for hand and head landmarks introduces some normalization issues, which we address.
2. These hand and head positions are now used as inputs to a Convolutional Recurrent Neural Network for classification. Labels are currently encoded using GloVe word embeddings, with multi-word phrases being encoded as the sum of their parts. This has much room for improvement.

## Current State
Many videos have been converted to sequences of hand/head locations and saved. A simple classification model has been built to predict the GloVe embedding of these samples, and has been trained for a handful of batches.

## TODOs
1. Dedicate the compute needed to convert more samples and train the baseline classifier.
2. Attempt to incorporate unsupervised pretraining by creating a hand-to-vec embedding model, to be used instead of raw hand and head coordinates.

## Implementations
```python

```
