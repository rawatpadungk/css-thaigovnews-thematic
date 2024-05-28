from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import json
from collections import defaultdict
from conversion import THAI_TO_ENG_TOPIC, FULLNAME_TO_ABBREVIAION

FIGSIZE = (12, 7)


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
    plt.figure(figsize=FIGSIZE)
    sns.displot(data=all_avg_scores)
    plt.title("Distribution of Scores")
    plt.xlabel("Average Sentiment Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()


def plot_sentiment_score_by_month():
    all_months = []
    score_by_month = []

    for year in os.listdir("sentiment_jsonl"):
        for month in os.listdir(os.path.join("sentiment_jsonl", year)):
            all_months.append(f"{year}-{month}")
            score_by_month.append(np.mean(get_sentiment_score_by_month(int(year), int(month))))

    df = pd.DataFrame({"month": all_months, "avg_score": score_by_month})
    save_path = "analysis/sentiment_score_by_month.png"
    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=df, x="month", y="avg_score")
    plt.title("Average Sentiment Score by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(range(len(all_months)), all_months, rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()


def get_frequency_and_avg_score_by_topic():
    topic_dict = defaultdict(lambda: {"frequency": 0, "avg_score": 0})
    for year in os.listdir("sentiment_jsonl"):
        for month in os.listdir(os.path.join("sentiment_jsonl", year)):
            for day in os.listdir(os.path.join("sentiment_jsonl", year, month)):
                open_path = os.path.join("sentiment_jsonl", year, month, day)
                with open(open_path, "r", encoding="utf8") as infile:
                    for line in infile:
                        data = json.loads(line)
                        converted_topic = THAI_TO_ENG_TOPIC.get(data["topic_model"], "Others")
                        topic_dict[converted_topic]["frequency"] += 1
                        topic_dict[converted_topic]["avg_score"] += data["avg_score"]
    for topic, detail in topic_dict.items():
        detail["avg_score"] /= detail["frequency"]

    return topic_dict


def plot_frequency_and_avg_score_by_topic():
    topic_dict = get_frequency_and_avg_score_by_topic()
    topics = list(topic_dict.keys())
    df = pd.DataFrame(
        {
            "topic": [FULLNAME_TO_ABBREVIAION.get(topic) for topic in topics],
            "frequency": [topic_dict[topic]["frequency"] for topic in topics],
            "avg_score": [topic_dict[topic]["avg_score"] for topic in topics],
        }
    )
    plt.figure(figsize=FIGSIZE)
    sns.barplot(data=df, x="topic", y="frequency", order=df.sort_values("frequency", ascending=False).topic)
    plt.title("Frequency by Topic")
    plt.xlabel("")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("analysis/frequency_by_topic.png")
    # plt.show()


if __name__ == "__main__":
    get_sentiment_score_distribution()
    plot_sentiment_score_by_month()
    plot_frequency_and_avg_score_by_topic()
