from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import json
from collections import defaultdict
from conversion import THAI_TO_ENG_TOPIC, FULLNAME_TO_ABBREVIAION

FIGSIZE = (12, 7)

# If possible, make it to a class object for good practice.


def get_sentiment_score(year: int, month: int):
    avg_score_one_month = []
    for day in os.listdir(os.path.join("sentiment_jsonl", str(year), str(month).zfill(2))):
        open_path = os.path.join("sentiment_jsonl", str(year), str(month).zfill(2), str(day))
        with open(open_path, "r", encoding="utf8") as infile:
            for line in infile:
                data = json.loads(line)
                avg_score_one_month.append(data["avg_score"])
    return avg_score_one_month


def get_sentiment_score_distribution(plot_central_tendency="median"):
    all_avg_scores = []
    if plot_central_tendency not in [None, "mean", "median", "mode"]:
        raise ValueError("central_tendency must be one of 'mean', 'median', or 'mode'")

    for year in os.listdir("sentiment_jsonl"):
        for month in os.listdir(os.path.join("sentiment_jsonl", year)):
            all_avg_scores.extend(get_sentiment_score(int(year), int(month)))

    stats = defaultdict(float)
    stats["mean"] = np.mean(all_avg_scores)
    stats["median"] = np.median(all_avg_scores)
    stats["mode"] = np.argmax(np.bincount(all_avg_scores))
    stats["std"] = np.std(all_avg_scores)
    stats["num_sample"] = len(all_avg_scores)
    central_score = stats.get(plot_central_tendency, None)

    plt.figure(figsize=FIGSIZE)
    sns.displot(data=all_avg_scores)
    if central_score:
        plt.axvline(central_score, color="red", linestyle="--", label=f"{plot_central_tendency} score")
        plt.legend()
    plt.title("Distribution of Scores")
    plt.xlabel("Average Sentiment Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("analysis/sentiment_score_distribution.png")
    # plt.show()

    return stats


def plot_sentiment_score():
    all_months = []
    score = []

    for year in os.listdir("sentiment_jsonl"):
        for month in os.listdir(os.path.join("sentiment_jsonl", year)):
            all_months.append(f"{year}-{month}")
            score.append(np.mean(get_sentiment_score(int(year), int(month))))

    df = pd.DataFrame({"month": all_months, "avg_score": score})
    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=df, x="month", y="avg_score")
    plt.title("Average Sentiment Score by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(range(len(all_months)), all_months, rotation=45)
    plt.tight_layout()
    plt.savefig("analysis/sentiment_score_by_month.png")
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


def plot_frequency_and_avg_score_by_topic(mean_score=None):
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
    sns.barplot(data=df, x="topic", y="frequency", order=df.sort_values("frequency", ascending=True).topic)
    plt.title("Frequency by Topic")
    plt.xlabel("")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("analysis/frequency_by_topic.png")
    # plt.show()

    # plt.figure(figsize=FIGSIZE)
    # sns.barplot(data=df, x="topic", y="avg_score", order=df.sort_values("avg_score", ascending=True).topic)
    # if mean_score:
    #     plt.axhline(mean_score, color="red", linestyle="--", label="mean score")
    #     plt.legend()
    # plt.title("Average Score by Topic")
    # plt.xlabel("")
    # plt.ylabel("Average Score")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig("analysis/avg_score_by_topic.png")
    # # plt.show()


def plot_score_by_topic():
    score_by_topic = defaultdict(list)
    for year in os.listdir("sentiment_jsonl"):
        for month in os.listdir(os.path.join("sentiment_jsonl", year)):
            for day in os.listdir(os.path.join("sentiment_jsonl", year, month)):
                open_path = os.path.join("sentiment_jsonl", year, month, day)
                with open(open_path, "r", encoding="utf8") as infile:
                    for line in infile:
                        data = json.loads(line)
                        converted_topic = THAI_TO_ENG_TOPIC.get(data["topic_model"], "Others")
                        score_by_topic[converted_topic].append(data["avg_score"])

    sorted_topic = sorted(score_by_topic, key=lambda x: np.mean(score_by_topic[x]))
    plt.figure(figsize=FIGSIZE)
    plt.boxplot(
        [score_by_topic[topic] for topic in sorted_topic],
        labels=[FULLNAME_TO_ABBREVIAION.get(topic) for topic in sorted_topic],
        patch_artist=True,
        showmeans=True,
        medianprops=dict(color="purple", linewidth=2.5),
        boxprops=dict(facecolor="yellow"),
        meanprops=dict(markerfacecolor="red", markeredgecolor="red", markersize=10),
    )
    plt.title("Sentiment Score by Topic")
    plt.xlabel("")
    plt.ylabel("Sentiment Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("analysis/sentiment_score_by_topic.png")
    # plt.show()


def get_sentiment_score_by_topic():
    all_months = []
    score_by_topic = defaultdict(list)
    for year in os.listdir("sentiment_jsonl"):
        for month in os.listdir(os.path.join("sentiment_jsonl", year)):
            score_by_topic_one_month = {eng_topic: [] for eng_topic in THAI_TO_ENG_TOPIC.values()}
            for day in os.listdir(os.path.join("sentiment_jsonl", year, month)):
                open_path = os.path.join("sentiment_jsonl", year, month, day)
                with open(open_path, "r", encoding="utf8") as infile:
                    for line in infile:
                        data = json.loads(line)
                        converted_topic = THAI_TO_ENG_TOPIC.get(data["topic_model"], "Others")
                        score_by_topic_one_month[converted_topic].append(data["avg_score"])
            for topic, scores in score_by_topic_one_month.items():
                score_by_topic[topic].append(np.mean(scores) if scores else np.nan)
            all_months.append(f"{year}-{month}")
    return score_by_topic, all_months


def get_sentiment_score_by_top_topic(n=5):
    topic_dict = get_frequency_and_avg_score_by_topic()
    sorted_topic_dict = sorted(topic_dict, key=lambda x: topic_dict[x]["avg_score"])

    top_positive_topics = sorted_topic_dict[-n:]
    top_negative_topics = sorted_topic_dict[:n]

    score_by_topic, all_months = get_sentiment_score_by_topic()

    df = pd.DataFrame(score_by_topic, index=all_months).interpolate()

    plt.figure(figsize=FIGSIZE)
    for topic in top_positive_topics:
        plt.plot(df.index, df[topic], label=topic)
    plt.legend(fontsize="small", loc="lower left")
    plt.title("Average Sentiment Score of Positive Topic by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(range(len(all_months)), all_months, rotation=45)
    plt.tight_layout()
    plt.savefig("analysis/sentiment_score_by_positive_topic_by_month.png")
    # plt.show()

    plt.figure(figsize=FIGSIZE)
    for topic in top_negative_topics:
        plt.plot(df.index, df[topic], label=topic)
    plt.legend(fontsize="small", loc="lower left")
    plt.title("Average Sentiment Score of Negative Topic by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(range(len(all_months)), all_months, rotation=45)
    plt.tight_layout()
    plt.savefig("analysis/sentiment_score_by_negative_topic_by_month.png")


def get_sentiment_score_frequent_topic(n=4):
    topic_dict = get_frequency_and_avg_score_by_topic()
    sorted_topic_dict = sorted(topic_dict, key=lambda x: topic_dict[x]["frequency"], reverse=True)
    frequent_topics = sorted_topic_dict[:n]

    score_by_topic, all_months = get_sentiment_score_by_topic()

    df = pd.DataFrame(score_by_topic, index=all_months).interpolate()

    plt.figure(figsize=FIGSIZE)
    for topic in frequent_topics:
        plt.plot(df.index, df[topic], label=topic)
    plt.legend()
    plt.title("Average Sentiment Score of Frequent Topic by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(range(len(all_months)), all_months, rotation=45)
    plt.tight_layout()
    plt.savefig("analysis/sentiment_score_by_frequent_topic_by_month.png")
    # plt.show()


if __name__ == "__main__":
    # stats = get_sentiment_score_distribution(plot_central_tendency="median")
    # plot_sentiment_score()
    # plot_frequency_and_avg_score_by_topic(mean_score=stats["mean"])
    # get_sentiment_score_by_top_topic()
    # get_sentiment_score_frequent_topic()
    plot_score_by_topic()
