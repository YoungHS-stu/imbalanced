from dataclasses import dataclass


@dataclass
class Result:
    id: int
    dataset: str
    resampler: str
    # resample_time: float
    trainer: str
    # train_time: float
    auc: float
    iba: float
    accuracy: float
    recall: float
    precision: float
    fscore: float
    gmean2: float
    # shuffle: bool = False
    original_maj_min: str = None
    resampled_maj_min: str = None
    resample_args: str = None
    train_args: str = None


if __name__ == "__main__":
    print("Result.py")
    id = 1
    dataset = "iris"
    resampler = "smote"
    resample_time = 0.1
    trainer = "svm"
    train_time = 0.1
    precision = 0.1
    recall = 0.1
    fscore = 0.1
    support = 1
    auc = 0.1
    gmean = 0.1
    shuffle = True
    # result = Result(id, dataset, resampler, resample_time, trainer, train_time, precision, recall, fscore, support, auc, gmean, shuffle)
    result = Result(
        id=id,
        dataset=dataset,
        resampler=resampler,
        resample_time=resample_time,
        trainer=trainer,
        train_time=train_time,
        precision=precision,
        recall=recall,
        fscore=fscore,
        support=support,
        # auc=auc,
        gmean=gmean,
        shuffle=shuffle
    )
    # print(result)
