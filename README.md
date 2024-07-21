Explainable Sparse Attention Mechanism
# Project Overview
This project aims to evaluate the effectiveness of a proposed sparse attention mechanism called "Syntactic Based Attention". This mechanism utilizes the prior information from a Dependency Parsing algorithm to constrain the attention mechanism in an explainable manner, rather than heuristically. Two experimental frameworks are implemented: Masked Language Modeling (MLM) and Natural Language Inference (NLI), using the MultiNLI dataset (a common sense dataset). The project compares the performance of a classic BERT-Base model with a custom version that incorporates the sparse attention mechanism.

## Experimental Frameworks
* Masked Language Modeling (MLM)
  - Task: Predicting masked words in a sentence.
  - Dataset: MultiNLI.
* Natural Language Inference (NLI)
  - Task: Determining if a given hypothesis is true (entailment), false (contradiction), or undetermined (neutral) based on a premise.
  - Dataset: MultiNLI.
## Models Compared
 * BERT-Base ModeL: A classic, widely used transformer model for natural language processing tasks.
 *  Custom Model which incorporates the proposed sparse attention mechanism.
## Key Findings
The Custom model showed only a 12% decrease in performance compared to the classic BERT-Base model in both MLM and NLI tasks.
The Custom model used significantly less contextual information: one third on average for the MLM task and one fifth on average for the NLI task.
These results underscore the "quality" of the "contextual" information considered by the Custom model, demonstrating the effectiveness of the sparse attention mechanism proposed.
## Requirements
Python 3.x
Libraries: TensorFlow or PyTorch, Transformers, scikit-learn, numpy, pandas, Spacy
