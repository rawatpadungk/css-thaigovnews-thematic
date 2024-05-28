from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import json


def get_sentiment_score_by_month(year: int, month: int):
    avg_score_one_month = []
    for day in os.listdir(os.path.join("sentiment_jsonl", str(year), str(month).zfill(2))):
        open_path = os.path.join("sentiment_jsonl", str(year), str(month).zfill(2), str(day))
        with open(open_path, "r", encoding="utf8") as infile:
            for line in infile:
                data = json.loads(line)
                avg_score_one_month.append(data["avg_score"])
    return avg_score_one_month


def get_sentiment_score_distribution():
    all_avg_scores = []
    for year in os.listdir("sentiment_jsonl"):
        for month in os.listdir(os.path.join("sentiment_jsonl", year)):
            all_avg_scores.extend(get_sentiment_score_by_month(int(year), int(month)))

    save_path = "analysis/sentiment_score_distribution.png"
    plt.figure(figsize=(10, 6))
    sns.displot(data=all_avg_scores)
    plt.title("Distribution of Scores")
    plt.xlabel("Average Sentiment Score")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.show()


def plot_sentiment_score_by_month():
    all_months = []
    score_by_month = []

    for year in os.listdir("sentiment_jsonl"):
        for month in os.listdir(os.path.join("sentiment_jsonl", year)):
            all_months.append(f"{year}-{month}")
            score_by_month.append(np.mean(get_sentiment_score_by_month(int(year), int(month))))

    df = pd.DataFrame({"month": all_months, "avg_score": score_by_month})
    save_path = "analysis/sentiment_score_by_month.png"
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="month", y="avg_score")
    plt.title("Average Sentiment Score by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(range(len(all_months)), all_months, rotation=45)
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    get_sentiment_score_distribution()
    plot_sentiment_score_by_month()
