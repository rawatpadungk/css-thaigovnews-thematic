from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os
import json


def get_avg_sentiment_score(year: int, month: int, day: str):
    avg_score_one_day = []
    open_path = os.path.join("sentiment_jsonl", str(year), str(month).zfill(2), str(day))
    with open(open_path, "r", encoding="utf8") as infile:
        for line in infile:
            data = json.loads(line)
            avg_score_one_day.append(data["avg_score"])
    return avg_score_one_day


def get_sentiment_score_distribution():
    all_avg_scores = []
    save_path = "analysis/sentiment_score_distribution.png"
    for year in os.listdir("sentiment_jsonl"):
        for month in os.listdir(os.path.join("sentiment_jsonl", year)):
            for day in os.listdir(os.path.join("sentiment_jsonl", year, month)):
                all_avg_scores.extend(get_avg_sentiment_score(int(year), int(month), day))

    plt.figure(figsize=(10, 6))
    sns.displot(data=all_avg_scores)
    plt.title("Distribution of Scores")
    plt.xlabel("Average Sentiment Score")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    get_sentiment_score_distribution()
