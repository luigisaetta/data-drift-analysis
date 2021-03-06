# Data Drift Analysis
In this repo I have collected all the work done, starting from Summer 2022, on tools to identify Data Drift

## What is Data Drift?
Let's imagine that you have a wonderful dataset, and you want to develop a Machine Learning model that enable to predict a target T,
giving the values for N features x1... xn.

> You have a good dataset, enough data, and you adopt a supervised approach. After some iterations (feature engineering, choice of the algorithm, hpo..)
you get good metrics (accuracy, f1-score, etc...) on the validation dataset and you're happy.
The next step is that someone in your team puts the model in production, for example as a REST service, and it is used by some nice applications to make its predictions.
Everyone, in your company is happy: it is a success story.

> But, after some time (months?) someone starts complaining that, having done some tests, **the predictions of your model are not as accurate as they were at the beginning**.
Performances (always measured on the above mentioned metrics) are starting to degradate.

This is called **Model Drift**. 

Your team start looking into it.

The first question you should make (and answer) is: are the data on which we're evaluation (scoring) our model "close" to the data used to train the model?
In other words: for every feature the mean, the variance, the occurrencies of different values (for categorical) are the same or the dataset has changed significantly?

If the dataset has changed we talk about **Data Drift**. 

And, this could be the main reason why the accuracy of your model is worse.

In this repository I have collected some Python tools that can be used to rapidly (well, depends on the dataset size, of course) identify if there are signals of Data Drift.

## Dynamical Models.
A ML model shouldn't be considered a static thing. 

From time to time, you should consider the option to re-train it on fresh data, in order to take into account the fact that data can change in time.

To diagnose a Model Drift is more complicated, because you need the expected values (the ground truth) to test your predictions. But, at least, you can periodically check to see if there are signs of Data Drift.

## Hypothesis Test.
We want to formalize the kind of test we're doing.

We have a first dataset. maybe the dataset on which the model has been trained. Or, in general, a reference dataset on which the model performs well.
We will call it a reference dataset.

You should always keep a good reference dataset.

Then, we have a second dataset. Maybe it is a new dataset, with data collected recently. And, we want to identify if there has been a Data Drift moving from the reference dataset to the new one.

At this point we make an Hypothesis. It is what is called the **"Null Hypotesis"**. Often identified with H0.

The "Null Hypothesys" is that dataset1 and dataset2 belong to the same Statistical Distribution.

We want, in some way, compute the probability of having the second dataset (better, the two datasets) if the Null Hypothesis is true.

If this probability (p_value) is too low, we can say that there is a strong signal of Data Drift.

Thresholds? It is always difficult to give values that can be used in any case. It depends on the size of the dataset. On the amount of "noise". It depends also on the "sensibility" that you want to give to your diagnostic tool.

But, often, the value 0.01 is used.

Therefore, a **quick recipe** could be:

```
if p_value < 0.01: 
   print("there is a drift")
```

## Kind of tests used.
In the dataset we have a set of features. Since we're only analyzing if there are signs of Data Drift, we don't need the values for the target T (the ground truth).

We will need first to identify a set of features that we want to consider in our test. For example, after we have validated our model, we do a "Feature Importance" Analysis. Then, we can decide to consider, let's say, the first 10 most important features.

We have then to make a difference between the **categorical features** and the **numerical (continuous)** features.

For the **categorical features** we will compute, for each of the distinct values the occurrencies and the frequencies.
Then, on top of this values, we can use a [**Chi2 Test**](https://en.wikipedia.org/wiki/Chi-squared_test).

For the **continuous features** we will use the [**Kolmogorov-Smirnov** test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test). 

And, we will also compute the [**Wasserstain**](https://en.wikipedia.org/wiki/Wasserstein_metric) distance.

The supporting functions can be taken from **scipy** library.

## It is not schema drift.
We're assuming here that the "schema" is the same for the two datasets.

What are changing are the Statistical characteristics of some key features (or example, the mean has increased significantly).

## Conda environment.
[Here](https://github.com/luigisaetta/data-drift-analysis/wiki/CONDA-Environments) you can find a list of the CONDA environments on which I have tested the code.

## JOBS and Data Drift Analysis in batch mode.
The code posted here can be used inside a Data Science JOB to do the analysis in batch mode and, therefore, can be included as part of a **ML pipeline**.

You will find more details [here](https://github.com/luigisaetta/data-drift-analysis/wiki/JOBS-for-Data-Drift-Analysis)

## Using the Model Catalog.
After you have trained your model, you should save it to the Data Science Model Catalog, using the [Model Serialization Framework](https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/user_guide/model_serialization/index.html)

For training your model, you have used a "reference" dataset, that has been splitted in train/validation/test. It is important to save this dataset, in order to have it as a "reference" to see if there is a Data Drift.

Assuming that the dataset has been stored in the Object Storage, you can use **Custom Metadata** in Model Catalog to save, together with the Model, the URI of the reference dataset file.

To give a working idea of what can be realized, I have created some Notebooks showing:
* how to store the model in the Catalog, together with the custom metadata "reference dataset"
* how to retrieve the "reference dataset url" from the Model Catalog.

In [this Notebook](https://github.com/luigisaetta/data-drift-analysis/blob/main/reference_dataset_from_model_catalog.ipynb) you will find a complete example showing how to load the dataset starting from the custom metadata saved in the Model Catalog.

More details in the Wiki page dedicated.

## The Wiki.
In the Wiki you can find some more details regarding the implementation.

The Wiki is WIP, therefore you'll find enough details as soon as I have enough time to document them.

