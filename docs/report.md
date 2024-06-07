# Next-word prediction project

**Student**: 11912007, Yahya JABARY

**Topic**: 3.2.3: Deep Learning for Image/Text Classification / Next-word prediction (Language Modelling) using Deep Learning

**Dataset**: https://github.com/GeorgeMcIntire/fake_real_news_dataset

<!--

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

---

The language modeling task is to assign a probability for the likelihood of a given word (or a sequence of words) to follow a sequence of words - as in $P(w_i | w_1, w_2, \ldots, w_{i-1})$.

There are many ways to approach this task, and over the years, we have come a long way from traditional markov chains, n-grams to recurrent neural networks and the latest transformer models[^fst]. The transformer model [^attention] not only is better at capturing long-range dependencies than its predecessors like LSTM [^lstm] but also GPU parallelizable and doesn't need any sequence unfolding to be trained which resolves the vanishing gradient problem and massively speeds up training.

---

---

[^fst]: History of natural language models: Jing, K., & Xu, J. (2019). A survey on neural network language models. arXiv preprint arXiv:1906.03591. https://arxiv.org/pdf/1906.03591
[^attention]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008). https://arxiv.org/pdf/1706.03762
[^lstm]: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780. https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735
