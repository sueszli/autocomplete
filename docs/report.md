# Next-word prediction project

**Student**: 11912007, Yahya JABARY

**Topic**: 3.2.3: Deep Learning for Image/Text Classification, Next-word prediction (Language Modelling) using Deep Learning

**Dataset**: Fake News Dataset - https://github.com/GeorgeMcIntire/fake_real_news_dataset

<!--

see: https://tuwel.tuwien.ac.at/course/view.php?id=64037

upload:

- zip file
- docker for reproducibility (explain why you chose this base image and the dependencies)
- markdown explaining how to run the code using CLI (argparse)
- report

report:

- your solutions (also describe failed approaches!)
- your results, evaluated on different datasets, parameters, ...
- an analysis of the results

testing:

- holdout dataset of not used sentences to test the completion
- automated or manual evaluation (subjective assessment)

-->

The language modeling task is to assign a probability for the likelihood of a given word (or a sequence of words) to follow a sequence of words, as in: $P(w_i | w_1, w_2, \ldots, w_{i-1})$.

In short, the goal is to predict the next word in a sentence given the previous words.

There are many ways to approach this task, and over the years, we have come a long way from traditional markov chains, n-grams to recurrent neural networks and the latest transformer models[^fst]. The transformer model [^attention] not only is better at capturing long-range dependencies than its predecessors like LSTM [^lstm] but also GPU parallelizable and doesn't need any sequence unfolding to be trained which resolves the vanishing gradient problem and massively speeds up training.

This recent breakthrough (among others) is the foundation of large scale language models like GPT-3, BERT (Bidirectional Encoder Representations from Transformers), and others.

This is a simple implementation of an LSTM based language model for educational purposes, trained on a dataset of fake and real news articles.

_Development Process_:

-   **Choosing PyTorch:** I was curious which framework RNNs are most elegant to be implemented in. So I tried both PyTorch and TensorFlow following several tutorials (see: `./playground/`). I decided to stick to PyTorch.
-   **Choosing an initial model:** Next following a tutorial on LSTM based language models [^kaggle] I implemented one in PyTorch. This model was initially trained on a dataset of Medium article titles and seemed to perform reasonably well right out of the box. It seemed like a good starting point.
-   **Choosing a dataset:** I modified the Kaggle model to run on multiple different datasets, in particular, a 0.5 GB Reddit dataset of comments from `r/jokes`. Due to the fact that it took too long to train on my local machine, I decided to pivot to a smaller dataset. I found an interesting dataset of fake and real news articles on GitHub.
-   **Optimizing:** I then optimized both the effectiveness and efficiency of the model by tuning hyperparameters and refactoring the code.

The model initially took 15min to train and had a K-Accuracy of 10.10%. But by the end of the development process, the model took only 10min to train and had a K-Accuracy of at least 16.91%.

Here is a nice visualization of the training process:

...

Finally I containerized the project using Docker for reproducibility.

---

[^fst]: History of natural language models: Jing, K., & Xu, J. (2019). A survey on neural network language models. arXiv preprint arXiv:1906.03591. https://arxiv.org/pdf/1906.03591
[^attention]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008). https://arxiv.org/pdf/1706.03762
[^lstm]: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780. https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735
[^kaggle]: See: https://www.kaggle.com/code/dota2player/next-word-prediction-with-lstm-pytorch/notebook
