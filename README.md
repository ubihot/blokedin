# Blokedin 0-1
Do you like to block people? I do! Let's block them with ML, just for fun.

We're gonna implement some ML techniques to block people based on their first message sent and evaluate which one performs better.


## Learning
This is a series of lectures to show what natural language processing (NLP) techniques are. We start simple from scratch and then build up to more complex architectures like Transformers and LLaMA.

This is for educational purpose only. We'll implement everything (almost) from scratch to learn machine learning techniques for natural language processing.

Aim is to understand how things work under the hood. We like the hood.

This is a supervised classification problem as we know the target when training the model and the target is a qualitative variable.

Maybe we create this with some assignments after the course is finished.


This course is also part of the Machine Learning course but covering the NLP side, some techinques are reusable.


## Setup

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch tensorflow matplotlib numpy pandas
```


## resources
- https://web.stanford.edu/~jurafsky/slp3/
- https://www.youtube.com/watch?v=Rvppog1HZJY (Stanford, funnily released on 2025-04-08)
- TODO include the CMU advanced course fall 2024
- TAD lectures slides
- TODO read and see if it helps https://course.fast.ai/ (it's not all NLP though) (maybe list it as a resource, )
- TODO write the history of nlp as in this video https://www.youtube.com/watch?v=Rvppog1HZJY
- TODO include any other useful resource here



## Lectures
The lectures are in the `notebooks` folder.



### Naive Bayes
vs random forest and support vector machine.
based on proability.

Pipeline to implement a naive bayes model.

- data collection -> it'll involve some processing to make faster and in the shape we want it
- Preprocess text -> token, remove punctuation.
- feature extraction -> Convert text to features. Bag-of-words (BOW), TF-IDF, or word embeddings (Word2Vec, GloVe, t-SNE <- should this be here?? I don't know what is t-SNE). For deep learning, maybe use embeddings directly. [I just copied this from the gpt]
- feature engineering -> Metadata like sender info, message length, time sent, etc.,
- model training -> 
- model evaluation ->
  - use confusion matrix, F1 score, precision accuracy measures for evaluation. recall. cross validation.
  - different experiments: remove url, keep url, replace url with a fixed token
  - AUC-ROC
  - Use cross-validation and a hold-out test set for reliable results

start with MultinomialNB as a baseline. What does this mean??

### Logistic Regression
Find the link between [features and the outcome](https://web.stanford.edu/~jurafsky/slp3/5.pdf).

In natural language processing, logistic regression is the baseline supervised machine learning algorithm for classification. Used to classify observation/s into one of two classes or one of many classes (**multinomial logistic regression**).

Logistic regression is a **discriminative classifier**. Similar to naive bayes, logistic regression is a probabilistic classifier.

logistic regression has two phases:
- **training**: TODO
- **test**: TODO


it seems like in logistic regression for classification we need a decision boundary.

we could do a sentiment classfication.


#### Advantages and disadvatages of Naive Bayes vs Logistic Regression

### K-Nearest-Neighbour


### Deep Learning (Neural Networs)

Start with a simple feedforward network using embeddings, then explore CNNs/RNNs. For best results, try fine-tuning pre-trained transformers (e.g., BERT, ELMo[what is elmo?? I don't know as of now, also what's google T5]). LSTM.

- implement BERT (what are bert alternatives?)
- implement a tokenizer in OCaml, either tiktoken or sentencepiece, reference the paper
- here we keep the whitespaces
- data imbalance: SMOTE, class weights, focal loss???
- what are the evaluation techniques
- Fine-Tune BERT for Spam Detection



### Ensemble/Hybrid Systems
- Combine deep learning with rule-based filters (e.g., regex patterns for phishing links) or metadata (sender info, message timing).
- Use model ensembles (e.g., stacking transformers with gradient-boosted trees).

### Advanced Techniques
- sft - do we have this in our example??
- checkpoint (not advanced though, it's just something to include)
- RLHF (paper)
- multi modal features: ???
- synthetic data?? and evaluation of all the models with or without synthetic data???
- a real world example with a real world deep learning framework like Tinygrad
- kind of implement the history of NLP, like gpt2, llama2, bert, elmo, naive bayes, n-grams, logistic regression, etc..
- ablation




Q: what are the frontier models as of now??