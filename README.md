# a part-of-speech tagger for identifiers

This implementation is a simple hmm trained on [this dataset](https://github.com/SCANL/datasets/blob/master/ensemble_tagger_training_data/training_data.csv)

the unlabeled identifiers came from the following open source repos and extracted with srcml
https://github.com/stanfordnlp/CoreNLP
https://github.com/elastic/elasticsearch
https://github.com/alibaba/fastjson
https://github.com/kdn251/interviews
https://github.com/TheAlgorithms/Java
https://github.com/iluwatar/java-design-patterns
https://github.com/SeleniumHQ/selenium
https://github.com/spring-projects/spring-boot

It uses the Viterbi algorithm to find the POS sequence with the highest probability for a given identifier

Inspired by the algorithm to use graph clustering for unsupervised POS tagging presented here https://aclanthology.org/P06-3002.pdf

This repo contains a few attempts to improve the accuracy of a HMM model to tag identifiers found in code by using the additional information contained in the code itself to provide hints at the correct tags.

The context of an identifier is one of `ATTRIBUTE, CLASS, DECLARATION, FUNCTION, PARAMETER`. This value is where the identifier was taken from in the code. When we don't pay attention to the context, we call it global.

The first attempt was to use a regular HMM that completely ignored the context (used the global context). This resulted in a model with an accuracy of 92%.

The next attempt we created a HMM that took into account the context. This was equivalent of having separate models for each context. This resulted in an accuracy of 94.9%.

Next we experimented with different weights between the global model and the context model for both the emission probabilities and transition probabilities. We found that we got a maximum accuracy of 95.1 with the following parameters.

|         | Emission weight | Transition weight |
|---------|-----------------|-------------------|
| Global  | 0               | 50                |
| Context | 100             | 50                |

Next we tried to extend the vocabulary of the model since we don't have a ton of training data. We did this by taking untagged identifiers and creating a weighted graph where each node is a word and each weight between word x and y is how many neighbors x shares with y. Then we used the chinese whispers graph clustering algorithm to identify clusters of related words, or words that could possibly be used interchangeably. Then we could search through each cluster for words that we know the emission probabilities of, and assign the emission probability of each word in the cluster to the probabilities of the known word. This resulted in an accuracy of 95%

Next we tried to use a similar approach with dense word embeddings. We trained a word2vec model on our untagged identifiers. Then we searched for words that were similar to known words. We could then extend the probabilities to the new words. This resulted in an accuracy of 93%

Both of these methods take a while to run since we have so many untagged identifiers.

Next we tried to augment our training data. We did this in 2 ways. The first way was to take a percent of our training data and switch the plurality of nouns and its corresponding tags. when running this on 100% of the identifiers we got an accuracy of 94%.

Next we tried to find synonyms for words in the training data that share a common pos tag. When running this on 100% of the identifiers we got an accuracy of 94%.

When we augment our training dat with both synonyms and plurality at 100% we get an accuracy of 93%.

Finally we added a brill-like post processing. We look for sequences of N, NPL, and NMs. Then we make all but the head noun NM. We also do the same with VM and V. When running on just nouns we get an accuracy of 94%. When running on just verbs we get 93%. And with both we get 93%.

There is more data in `weight.xlsx`
There is example output in `run.out`

Note, if running on an onDemand environment, there are a few packages missing, `inflect`, `networkx`, and `chinese_whispers`
To install these packages locally run `pip3 install inflect networkx chinese_whispers --user`
These packages are only used for graph clustering and post processing. in order to run without building the models for graph clustering, use `python3 src/main ondemand`. This will use the pre-generated augmented models for evaluation.

To run without building the word2vec model and graph clustering: `python3 src/main`
To run and build the word2vec model and graph clustering `python3 src/main long`
To run without libraries available on ondemand use `python3 src/main ondemand`
To start a webserver run `cd src`, `export FLASK_APP=webserver` and `flask run`. Note flask needs to be installed with `pip3 install flask` (I don't know if there is a way to install it as user)

---

To run the RNN portion of this project, open the included 'RNN_POS_TAGGER.ipynb' in a Google Colab environment and follow the instructions included in the project. Specifically, the first line of "Usage" stating that the "easiest way to the script by default is to open 'Runtime' in the topbar in Google Colab and then select 'Run all'."

