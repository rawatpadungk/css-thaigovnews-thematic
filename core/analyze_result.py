import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from conversion import THAI_TO_ENG_TOPIC, FULLNAME_TO_ABBREVIAION

FIGSIZE_RECT = (12.5, 6.5)
FIGSIZE_SQR = (10, 10)

random.seed(42)


class Analysis:
    """A class to analyze the sentiment data."""

    def __init__(self):
        self.all_months = self.get_all_months()
        self.all_topics = self.get_all_topics()
        self.monthly_score_by_topic = self.get_monthly_score_by_topic()
        self.all_scores = self.get_all_scores()
        os.makedirs("visual_analysis", exist_ok=True)

    def plot_all_visualizations(self):
        """Plot all visualizations including the barplot, histplot, boxplot, and lineplot."""
        self.barplot_freq_by_topic()
        self.histplot_score_distribution()
        self.boxplot_score_by_topic()
        for specific_type in ["frequent", "positive", "negative"]:
            self.lineplot_monthly_score_by_specific_topics(specific_type=specific_type)

    def get_all_months(self):
        """Return all months in the sentiment_jsonl directory."""
        all_months = []
        for year in os.listdir("sentiment_jsonl"):
            for month in os.listdir(os.path.join("sentiment_jsonl", year)):
                all_months.append(f"{year}-{month}")
        return all_months

    def get_all_topics(self):
        """Return all topics in the sentiment_jsonl directory."""
        return FULLNAME_TO_ABBREVIAION.keys()

    def get_monthly_score_by_topic(self):
        """Return the monthly score by topic."""
        monthly_score_by_topic = defaultdict(lambda: {topic: [] for topic in self.all_topics})
        # Iterate through the sentiment_jsonl directory and store the data in the dataframe format
        for year in os.listdir("sentiment_jsonl"):
            for month in os.listdir(os.path.join("sentiment_jsonl", year)):
                for day in os.listdir(os.path.join("sentiment_jsonl", year, month)):
                    open_path = os.path.join("sentiment_jsonl", year, month, day)
                    with open(open_path, "r", encoding="utf8") as infile:
                        for line in infile:
                            data = json.loads(line)
                            year_month = f"{year}-{month}"
                            converted_topic = THAI_TO_ENG_TOPIC.get(data["topic_model"], "Others")
                            monthly_score_by_topic[year_month][converted_topic].append(data["avg_score"])
        return pd.DataFrame(monthly_score_by_topic).T

    def get_avg_monthly_score_by_topic(self):
        """Return the average monthly score by topic."""
        return self.monthly_score_by_topic.map(np.mean)

    def get_monthly_freq_by_topic(self):
        """Return the monthly frequency by topic."""
        return self.monthly_score_by_topic.map(len)

    def get_all_scores(self):
        """Return all scores in the sentiment_jsonl directory."""
        all_scores = sum(self.monthly_score_by_topic.values.flatten(), [])
        self.mean_score = np.mean(all_scores)
        self.median_score = np.median(all_scores)
        return all_scores

    def barplot_freq_by_topic(self):
        """Plot the frequency by topic."""
        # Plot the barplot of the frequency by topic and sort the values
        df = self.get_monthly_freq_by_topic().sum(axis=0).sort_values()
        self.frequent_topics = df.index[-4:]
        plt.figure(figsize=FIGSIZE_SQR)
        sns.barplot(x=[FULLNAME_TO_ABBREVIAION.get(topic) for topic in df.index], y=df.values, color="#029386")
        plt.rcParams.update({"font.size": 26})
        plt.title("Frequency by Topic")
        plt.xlabel("")
        plt.ylabel("Frequency", fontsize=22)
        plt.xticks(rotation=90, fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig("visual_analysis/frequency_by_topic.png")

    def histplot_score_distribution(self):
        """Plot the distribution of scores."""
        # Plot the histogram of the scores and add the vertical line for the median score
        plt.figure(figsize=FIGSIZE_SQR)
        sns.histplot(self.all_scores, bins=20, color="orange")
        plt.axvline(self.median_score, color="red", linestyle="--", label="median score")
        plt.legend()
        plt.rcParams.update({"font.size": 26})
        plt.title("Distribution of Scores")
        plt.xlabel("Average Sentiment Score")
        plt.ylabel("Frequency")
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig("visual_analysis/sentiment_score_distribution.png")

    def boxplot_score_by_topic(self):
        """Plot the sentiment score by topic."""
        # Plot the boxplot of the scores by topic and sort the values
        score_by_topic = self.monthly_score_by_topic.apply(lambda col: sum(col, []), axis=0)
        sort_idx = score_by_topic.apply(np.mean).sort_values().index
        self.positive_topics = sort_idx[-4:]
        self.negative_topics = sort_idx[:4]
        df = score_by_topic[sort_idx]
        plt.figure(figsize=FIGSIZE_RECT)
        plt.rcParams.update({"font.size": 16})
        plt.boxplot(
            df,
            labels=[FULLNAME_TO_ABBREVIAION.get(topic) for topic in df.index],
            patch_artist=True,
            showmeans=True,
            boxprops=dict(facecolor="lightblue"),
            medianprops=dict(color="black", linewidth=2.5),
            meanprops=dict(markerfacecolor="#E50000", markeredgecolor="#E50000", markersize=10),
        )
        plt.title("Sentiment Score by Topic")
        plt.xlabel("")
        plt.ylabel("Sentiment Score")
        plt.xticks(rotation=75, fontsize=15)
        plt.tight_layout()
        plt.savefig("visual_analysis/score_by_topic.png")

    def lineplot_monthly_score_by_specific_topics(self, specific_type=None):
        """Plot the trend of the average monthly score by specific topics."""
        # Determine the specific topics to plot based on the specific type
        if specific_type == "frequent":
            specific_topics = list(self.frequent_topics).copy()
            specific_topics.remove("Government News")
        elif specific_type == "positive":
            specific_topics = self.positive_topics
        elif specific_type == "negative":
            specific_topics = self.negative_topics
        else:
            specific_topics = self.all_topics
        # First, plot the overall monthly score and the horizontal line for the mean score
        avg_monthly_score_by_topic = self.get_avg_monthly_score_by_topic()
        avg_monthly_score = self.monthly_score_by_topic.apply(lambda row: sum(row, []), axis=1).apply(np.mean)
        plt.figure(figsize=FIGSIZE_RECT)
        plt.rcParams.update({"font.size": 16})
        plt.axhline(self.mean_score, color="grey", linestyle="--", label="mean score")
        sns.lineplot(
            x=avg_monthly_score.index,
            y=avg_monthly_score.values,
            label="avg monthly score",
            linewidth=2.5,
            color="black",
            linestyle="--",
        )
        # Then, interpolate the missing value and consider only the specific topics
        # Plot the lineplot of the average monthly score of each specific topic
        df = avg_monthly_score_by_topic.interpolate().loc[:, specific_topics]
        colors = ["red", "#FFA500", "green", "#0077B6"]
        for i, topic in enumerate(df.columns):
            sns.lineplot(x=df.index, y=df[topic], color=colors[i], label=topic)
        plt.legend(fontsize=14, ncol=2, loc="lower left")
        plt.title(f"Average Monthly Sentiment Score of {specific_type.title} Topic")
        plt.xlabel("")
        plt.ylabel("Average Sentiment Score", fontsize=15)
        plt.xticks(range(len(self.all_months)), self.all_months, rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(f"visual_analysis/monthly_score_by_{specific_type}_topic.png")


if __name__ == "__main__":
    Analysis().plot_all_visualizations()
