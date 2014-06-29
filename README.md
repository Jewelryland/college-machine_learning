Sentiment Analysis (Movie Reviews)
==================================

This code classifies movie reviews.

Data
----

Inside the `data/` folder, there are movie reviews, which are separated into
positive and negative reviews. This data has been taken from an [online
database](https://www.cs.cornell.edu/people/pabo/movie-review-data/).

There are also words, which are split into positive and negative. Later
the python files generate closely selected subset of words, which are then
used for better classification.

Installation
------------

This project uses Python 2.7. Necessary packages are:

* `numpy` (requires Fotran to be installed)
* `nltk`
* `scikit-learn`

If you're on Linux or Mac, all of these packages should be available with a
simple `pip install`.

Usage
-----

### `basic.py`

```sh
$ python basic.py
```

This will classify movie reviews using Naive Bayes, and 2 variants of SVM
(with linear and polynomial kernel). It will train the classifiers
and print

* the accuracy of the classifiers
* 2 simple examples of movie reviews, and show what do our classifiers classify
  them to (positive or negative)
* a confusion matrix + precision, recall and F score

You can choose whether you'll use all positive and negative words, or just
more specific/helpful for the reviews, by setting the `word_list` variable
in the top of the file.

### `parameters.py`

```sh
$ python parameters.py
```

It contains code (unstable) for finding the right parameters for SVM.

### `select_words_frequency.py`

```sh
$ python select_words_frequency.py
```

It analizes the list of words (positive and negative) that we already have,
and selectes by frequency and TF-IDF, so that our classifiers can choose
words which are more relevant to the actual reviews. It fills in
`data/words/selected-words-frequency.txt` and `data/words/selected-tfidf.txt`

### `lib/`

It contains the core logic. It loads appropriate words, and trains on them
and does simple or k-fold cross validation. It has two classifiers: Naive Bayes
and SVM (3 variants).

