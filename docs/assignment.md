see: https://tuwel.tuwien.ac.at/course/view.php?id=64037

deadline: 24h before the discussion slot you registered for

chosen topic: 3.2.1: Deep Learning for Image/Text Classification / Next-word prediction (Language Modelling) using Deep Learning

-   given an incomplete sequence of words, find the most likely next word
-   use deep learning

requirements:

-   implement a CLI
-   make project reproducible
-   add documentation that explains usage
-   upload a zipped file
-   upload a report explaining: approach, failures and solutions, results evaluated on different datasets and parameters
-   you can use existing models/architectures and tune them

evaluation:

-   keep a holdout dataset of not used sentences to test the completion
-   text is ambiguous, and multiple words would be acceptable as “correct” next word. Thus, also consider how to evaluate your model; automated methods are fine, as well as manual / subjective assessment.
-   compare not just the overall measures, but perform a detailed comparison and analysis per class (confusion matrix), to identify if the two approaches lead to different types of errors in the different classes, and also try to identify other patterns.

datasets:

-   fake news dataset: https://github.com/GeorgeMcIntire/fake_real_news_dataset
