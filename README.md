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
- include the CMU advanced course fall 2024
- TAD lectures slides
- TODO read and see if it helps https://course.fast.ai/ (it's not all NLP though) (maybe list it as a resource, )
- TODO write the history of nlp as in this video https://www.youtube.com/watch?v=Rvppog1HZJY


## Intro Stuff
mainly following the Speech and Language Processing book
- Tokenization
- Regular Expressions
- Edit Distance
- BPE, what was before BPE?? what's the BPE paper? include it here

#### 
ELIZA is a simple pattern recognition program (chatbot) that acts like a Rogerian psycotherapist in the 1966, acted like a listener that knows nothing about the world.

One tool to describe text pattern is **regular expression**, for example to extract strings from document.
**text normalization** means converting it to a more convenient standard form.
**tokenization** separating words or word parts from the text document, e.g. english words are separated by whitespace (not always sufficient).
for tweets for example we'd need to tokenize the **emoticons** :) or **hashtag** #nlp.
some languages are hard to tokenize as they don't have spaces, e.g. japanese.
we'd need sometimes to tokenize subwords, short phrases, letters for large LLMs.
another part of text normalization is **lemmatization**, the task of determining that words have the same root, e.g. *sang*, *sung*, *sings* are form of the verb *sing*. *sing* is the common *lemma* of these words. A **lemmatizer** maps from all these to *sing*.
**stemming** is a simpler form of lemmatization in which we just strip suffixes from the end of the word.
Text normalization also includes **sentence segmentation**, breaking text into sentences using cues like periods or exclamation.
The metric **edit distance** measures how similar two strings are based on number of edits (insertion, deletion, substitution) it takes to change one string into another.

#### Regular Expressions
regex is a language for specifying text search strings, used e.g. in `grep`, `vim`.
very useful when we have a **pattern** to search for in a **corpus** to search through.
a regex will search through the corpus returning all the pattern matches.
the corpus can be a single document or a collection.
we'll describe the **extended regular expressions**.

regular expression patterns are case sensitive (refer to `./notebooks/lecture_1.ipynb` for more on the actual use case of these):
- concatenation: putting characters in sequence is concatenation, e.g. `/woodchucks/`, `/ubaid/`
- range: using the `[-]` (use of [] is required) to indicate from to a range, e.g. `/[A-Z]/`, `/[0-9]/`, `/[a-z]/`
- Kleene*: * to say how many of something, means zero or more occurrences
- Kleene+: at least one
- wildcard: `.` matches any one character
- anchors: anchor the regular expressions to a particular place in a string
  - `^`   start of line
  - `$`   end of line
  - `\b`  word boundary
  - '\B`  non-word boundary
- **disjunction** operator: `|` specifies either or, e.g. `dog|cat`
- enclosing the sequences with `()` we basically make it like a single character
regualar expressions are **greedy** (match as large as possible) by default but we can use **non-greedy** Kleene operators (match as little as possible). `*?` or `+?`

an example is trying to match all the words `the` in a document
we start with /the/ but we're missing beginning of text for example `The`
so we do this /[Tt]he/
but that we'll also match `other` `there`
so we need word boundary like /\b[Tt]he\b/


^ the above proccess introduces **false positives**, i.e. matching `other` `there`
and also false negatives, i.e. missing correct strings

reducing the overall error of the application involves two antagonists efforts:
- increasing **precision** (minimizing false positives)
- increasing **recall** (minimizing false negatives)


an important part of regular expression is in **substitutions**. e.g. in python or in vim the sub operator like `s/colour/color`

you can use the number operator \1 to refer to a matched pattern back.
this use of parenthese to store a pattern in memory is called **capture group**, every time a capture group is used, the resulting match is stored in a numbered **register**. And you refer to the captured group via numbers like 1, 2, 3, etc..

if we want to use parenthese to not capture the group we can use a non-capturing group via /(?:some|a few) (people|cat) like some \1/
in the example above `\1` would refer to the second parenthese group, i.e. `(people|cat)`

Recall ELIZA, that is using a series of cascade regular expression substitution.
e.g. some are these
```txt
s/.* YOU ARE (depressed|sad) .*/I AM SORRY TO HEAR YOU ARE \1/
s/.* YOU ARE (depressed|sad) .*/WHY DO YOU THINK YOU ARE \1/
s/.* all .*/IN WHAT WAY/
s/.* always .*/CAN YOU THINK OF A SPECIFIC EXAMPLE/
```

##### Words
what does count as a word? we can decide to treat punctuation as a separate word or not depending on the task, e.g. part of speech tagging.
**utterrance** is the spoken correlate of a sentence. 
e.g. "I do uh main- mainly business data processing"
there are two kinds **disfluencies**. `main-` broken word is called **fragment**
where uh um are called **fillers** or **filled-pauses**. should we consider these as words? depends on the application.

to understand better what counts as a word we need to understand **word types**, i.e. number of distinct words in the corpus.
word **instances** are the total number of N of running words, equivalent of word tokens in the past.

do we consider `They` and `they` as two word types or the same? it depends on the task. e.g. for speech recognition same is fine.

the relantionship between the word type and the word instance is referred to by **Herdan's law** or **Heaps' law**.

cats and cat are two differenct **wordforms** but have the same **lemma**. A lemma  is a set of lexical forms having the same stem. The **wordform** is the full inflected or derived form of the word. 

for many LLMs we actually use **tokens** using the **tokenization** process. The token can be a word or a part of the word.

##### Corpora
there are variations genre of text. e.g. from telephone conversations, business meetings, medical interviews, etc..
to understand what a corpus was meant for is thanks to **datasheet** or **data statements** that includes:
- motivation for collecting the corpus
- situation: when and in what situation was text written/spoken
- language variety: what language was the corpus in?
- speaker demographics: what was e.g the age, sex of the text's authors?
- collection process: how big is the data? if it is a subsample how was it sampled? was the data collected with consent? how was the data preprocessed? and what metadata is available?
- annotation process: what are the annotations, how was the data annotated? how was the annotation process?
- distribution: are there copyright or other intellectual property restrictions?


##### Text Normalization
before any natural language processing of a text, the text has to be normalized through the **text normalization** process which involves:
- tokenization (segmentation) words
- normalizing word formats
- segmenting sentences


NOTE: ok at this point we can start collecting the data and do some of text normalization process in a jupyter notebook.


**NOTE what happens when you have a corpus with lots of examples of one label??? we need to explain it in our lectures**

**the other thing we need to deal with the preprocessing or processing of our corpus for our task is the handling of links, we could potentially neglect them, i.e. remove them with a regular expression so we basically showcase the use of how regex are used, we could do that in one type of classification or in the other one we could just keep it or replace with a placeholder like [LINK], we need to keep the script so that it has both versions, for now we don't do anything special.**

- For the above mentioned we should compare the performance of the different solutions.
- Create features engineers.
- Create synthetic data after training the model.
- We could also have one model trained by giving the subject as one of the features.
- We've also included the has_attachment feature and the subject.
- Ideally also when we create the synthetic data we want to be able to generate these features too so in the lecture descriptions we can explain how the original data looked like.
- We should train with capital or lowercase only and se the diffrence and show the difference.**

**how do we deal with the fact that our corpus doesn't have all the words that exists in english vocabulary???**


##### Word and Subword Tokenization
In NLP, we usally break words into **subword tokens**, which can be words or part of words or individual letters.

Tokenization is run before any other language processing.

**Top Down (Rule Based) Tokenization**
In NLP, we usually keep the punctuations and numbers. Then we need to account for hashtags, urls, emails, dates, special chars in words like AT&T, prices (e.g. $45.45), etc...

Can use tokenizer to expand **clitic** contractions marked by apstrophes, e.g. `what're` into `what are`.
A clitic is a part of word that cannot stand on its own.

Tokenization is tied with **named entity recognition**, task of detecting names, dates and organizations.

A common tokenization standard is the **Penn Treebank Tokenization**, where `doesn't` becomes `does n't`.

Word tokenization is more comlex in Chinese and Thai where there are no spaces and the words are composed of characters called **hanzi** (Chinese).
Each character represents a single unit of meaning (**morpheme**).

For some languages like thai we need more than one character to use as a word.

**Byte-Pair Encoding: A Bottom Up Tokenization Algorithm**
We can use the data to tell us what the words should be unlike the previous approach where we either used a character or whitespace or sth more complex. This is very useful to deal with unknown words, very common in NLP.
In NLP, algos learn facts from one corpus (**training** corpus) and use theses facts to make decisions about separate **test** corpus, hence the problem of unknown words.

To solve this problem, tokenizers try to induce **subwords** tokens.

Most tokenizer scheme have 2 parts:
- **token learner**: takes raw training corpus and induces a vocabulary, a set of tokens.
- **token segmenter**: takes raw test sentence and segments it into the tokens vocabulary.

Two algorithms are used:
- **byte pair encoding** (Sennrich et al., 2016)
- **unigram language model** (Kudo, 2018)

**SentencePiece** (Kudo and Richardson, 2018a) has both implementation, but SentencePiece is usually referred to mean as **unigram language model**.


###### BPE
- begin with a vocabulary that is all the individual characters.
- examine training corpus
- choose the two symbols that are most frequently adjacent, e.g. 'A' and 'B'
- add the merged symbol 'AB' to the vocabulary
- replace every adjacent 'A' 'B' with 'AB'
- continue to count and merge creating longer and longer character strings, until k merges have been creating k novel tokens
- k thus becomes the parameter of the algorithm
- the resulting vocabulary consists of the original characters plus k new symbols.

TODO
- implement the BPE algorithm in ocaml
- go through the karpathy implementation of BPE in its video https://youtu.be/zduSFxRajkE


##### Word Normalization, Lemmatization and Stemming
The simplest case of word normalization is **case folding**. e.g. mapping everything to lowercase like `Woodchuck` and `woodchuck` are represented identically, which is very helpful for generalization in tasks like information retrieval or speech recognition.

For sentiment analysis and other text classification taks, information extraction, machine translation instead case folding is generally not done.

If you use BPE you may not need to do any other normalization.


**lemmatization**
is the task of determining that two words have the same root. e.g. one application of it could be `He is reading detective stories` -> `He be read detective story`

How is lemmatization done?
**morphology** is the study of the of the way words are built up from smaller meaning-bearing units falled **morphemes**.
**stem** is the central morpheme of the word, gives the main meaning
**affixes** adding additional meanings of various kind.
e.g. fox is one morpheme; cats is two `cat` and `s`

Lemmatization algos can be complex, hence we can use a simpler morphological analysis called **stemming**, i.e. chopping off the final affixes.
**Porter Stemming** consists of rewrite rules run in a series. Not commonly used now cause of overgeneralization and undergeneralization.

##### Sentence Segmentation
Usually by using punctuation, e.g. periods, question marks, exclamation. Period is more ambiguous.
Sentence tokenization woks by deciding (machine learning or rule) whether period is part of word or is sentence boundary marker, an abbreviation dictionary can help find abbreviations.


##### Minimun Edit Distance
In NLP one common task is to measure how similar two strings are, e.g. graffe with giraffe.
Another example is **coreference**, i.e. decide whether two strings refers to the same entity.
e.g.
```
Stanford Arizona Cactus Garden.
Stanford University Arizona Cactus Garden.
```
another task of strings similarity is in the quality measure of transcription produced by a speech recognition system, words that differ by a lot have worse quality transcription and those that differ by a few have better quality.

**Edit distance** gives us the technique to quantify these intuitions about similarity.
**Minimum edit distance** is defined as the minimum number of editing operations (insertions, deletions, substitutions) needed to transform one string into another.
We can even assign even a cost to these operations when doing alignment.
The **Levenshtein** distance between two sequences is in which each of these three operations has a cost of 1.



##### Summary
we covered
- regular expression: a powerful tool for pattern matching
- **concatenation** of symbols, **disjunction** ([], |), **counters** (*, +, {n,m}), **anchors** (^, $) and precedence operators ((,)).
- **word tokenization and normalization** 
- **Porter** simplest algorithm for stemming
- **minimum edit distance** using **dynamic programming** and **alignment** of two strings






### Regular Expressions, Tokenization, Edit Distance (maybe/?)
ELIZA was a chatbot like program using pattern recognition phrases like "I need X" and translate into suitable output like "what would you do if you got X?". It didn't know anything about the world, it was like a listener that acts like knowing nothing.

One of the most important tool to describe text.



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