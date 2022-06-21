# Data Drift Analysis
In this repo I have collected all te work done, starting from summer 2022, on tools to identify data drift

## What is Data Drift?
Let's imagine that we have a worderful dataset, and we want to develop a Machine Learning model that enable us to predict a target T,
giving the values for N features x1... xn.

We have a good dataset, we have enough data, we adopt a supervised approach. After some iterations (feature engineering, choice of the algorithm, hpo..)
we get good metrics (accuracy, f1-score, etc...) on the validation dataset and we're happy.
The next step is that someone in your team puts the model in production, for example as a REST service, and it is used by some nice applications to make its predictions.
Everyone, in your company is happy: it is a success story.

But, after some time (months, years?) someone starts complaining that, having done some tests, **the predictions of your model are not as accurate as they were at the beginning**.
Performances (always measured on the above mentioned metrics) are starting to degradate.

This is called **Model Drift**. 

Your team start looking into it.

The first question you should make (and answer) is: are the data on which we're evaluation (scoring) our model "close" to the data used to train the model?
In other words: for every feature the mean, the variance, the occurrencies of different values (for categorical) are the same or the dataset has changed significantly?

If the dataset has changed we talk about **Data Drift**. 

And, this could be the main reason why the accuracy of your model is worse.

In this repository I have collected some Python tools that can be used to rapidly (well, depends on the dataset size, of course) identify if there are signals of Data Drift.

## Hypothesis Test.
We want to formalize the kind of test we're doing.

We have a first dataset. maybe the dataset on which the model has been trained. Or, in general, a reference dataset on which the model performs well.
We will call it a reference dataset.

You should always keep a good reference dataset.

Then, you have a second dataset. Maybe it is a new dataset, with data collected recently. And, we want to identify if there has been a Data Drift moving from the reference dataset to the new one.

At this point we make an Hypothesis. It is what is called the **"Null Hypotesis"**. Often identified with H0.

The "Null Hypothesys" is that dataset1 and dataset2 belong to the same Statistical Distribution.

We want, in some way, compute the probability of having the second dataset (better, the two datasets) if the Null Hypothesis is true.

If this probability (p_value) is too low, we can say that there is a strong signal of Data Drift.

Thresholds? Well, it is always difficult to give values that can be used in any case. It depends on the size of the dataset. On the amount of "noise". It depends also on the "sensibility" that you want to give to your diagnostic tool.

But, often, the value 0.01 is used.

Therefore, a quick recipe could be:

if p_value < 0.01: "there is a drift"




