# a part-of-speech tagger for identifiers

This implementation is a simple hmm trained on [this dataset](https://github.com/SCANL/datasets/blob/master/ensemble_tagger_training_data/training_data.csv)

It uses the Viterbi algorithem to find the POS sequence with the highest probablility for a given identifier

To run: `python3 src/generate_probs.py`
