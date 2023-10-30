# TODO: In this cell, write your BaselineChallenge flow in the baseline_challenge.py file.
from collections import Counter

from sklearn.metrics import accuracy_score, roc_auc_score

from metaflow import (
    FlowSpec,
    step,
    Flow,
    current,
    Parameter,
    IncludeFile,
    card,
    current,
)
from metaflow.cards import Table, Markdown, Artifact, Image
import numpy as np
from dataclasses import dataclass

def labeling_function(row):
    """
    A function to derive labels from the user's review data.
    This could use many variables, or just one.
    In supervised learning scenarios, this is a very important part of determining what the machine learns!

    A subset of variables in the e-commerce fashion review dataset to consider for labels you could use in ML tasks include:
        # rating: Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
        # recommended_ind: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.
        # positive_feedback_count: Positive Integer documenting the number of other customers who found this review positive.

    In this case, we are doing sentiment analysis.
    To keep things simple, we use the rating only, and return a binary positive or negative sentiment score based on an arbitrarty cutoff.
    """
    # TODO: Add your logic for the labelling function here
    # It is up to you on what value to choose as the cut off point for the postive class
    # A good value to start would be 4
    # This function should return either a 0 or 1 depending on the rating of a particular row

    POSITIVE_MIN_SCORE = 4
    if row.rating >= POSITIVE_MIN_SCORE:
        label = 1
    else:
        label = 0
    
    return label


@dataclass
class ModelResult:
    "A custom struct for storing model evaluation results."
    name: None
    params: None
    pathspec: None
    acc: None
    rocauc: None


class BaselineChallenge(FlowSpec):
    split_size = Parameter("split-sz", default=0.2)
    data = IncludeFile("data", default="Womens Clothing E-Commerce Reviews.csv")
    kfold = Parameter("k", default=5)
    scoring = Parameter("scoring", default="accuracy")

    @step
    def start(self):
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = self.data  #IncludeFile("data", default="Womens Clothing E-Commerce Reviews.csv")
        # Look up a few lines to the IncludeFile('data', default='Womens Clothing E-Commerce Reviews.csv').
        # You can find documentation on IncludeFile here: https://docs.metaflow.org/scaling/data#data-in-local-files

        # filter down to reviews and labels
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df = df[~df.review_text.isna()]
        df["review"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        reviews = _has_review_df["review_text"]
        labels = _has_review_df.apply(labeling_function, axis=1)
        self.df = pd.DataFrame({"label": labels, **_has_review_df})

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({"review": reviews, "label": labels})
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f"num of rows in train set: {self.traindf.shape[0]}")
        print(f"num of rows in validation set: {self.valdf.shape[0]}")

        self.next(self.baseline, self.model)

    @step
    def baseline(self):
        "Compute the baseline"

        from sklearn.metrics import accuracy_score, roc_auc_score

        self._name = "baseline"
        params = "Always predict 1"
        pathspec = f"{current.flow_name}/{current.run_id}/{current.step_name}/{current.task_id}"

        # TODO: predict the majority class
        class_counts = Counter(self.traindf.label)
        keys = list(class_counts.keys())
        majority_class = keys[np.argmax(class_counts.values())]
        predictions = np.full(len(self.valdf), majority_class, dtype=np.int64)

        acc = accuracy_score(self.valdf.label, predictions)
        rocauc = roc_auc_score(self.valdf.label, predictions)
        self.result = ModelResult("Baseline", params, pathspec, acc, rocauc)
        self.next(self.aggregate)

    @step
    def model(self):
        from model import NbowModel

        self._name = "model"
        # NOTE: If you followed the link above to find a custom model implementation,
        # you will have noticed your model's vocab_sz hyperparameter.
        # Too big of vocab_sz causes an error. Can you explain why?
        self.hyperparam_set = [{"vocab_sz": 100}, {"vocab_sz": 300}, {"vocab_sz": 500}]
        pathspec = f"{current.flow_name}/{current.run_id}/{current.step_name}/{current.task_id}"

        self.results = []
        for params in self.hyperparam_set:
            model = NbowModel(params["vocab_sz"])
            model.fit(X=self.traindf["review"], y=self.traindf["label"])
            predictions = model.predict(self.valdf["review"])

            acc = accuracy_score(self.valdf["label"], predictions)
            rocauc = roc_auc_score(self.valdf["label"], predictions)
            self.results.append(
                ModelResult(
                    f"NbowModel - vocab_sz: {params['vocab_sz']}",
                    params,
                    pathspec,
                    acc,
                    rocauc,
                )
            )

        self.next(self.aggregate)

    @step
    def aggregate(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    BaselineChallenge()
