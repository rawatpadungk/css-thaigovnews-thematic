from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import json
from collections import defaultdict


def convert_thai_topic_to_eng(topic: str):
    rule_based_conversion = {
        "Cabinet Meeting Synopsis": "Cabinet Meeting Synopsis",
        "Government News": "Government House News",
        "Statements": "Statements",
        "": "Others",
        "ข่าวคณะโฆษก": "Government Spokesperson News",
        "ข่าวนายกรัฐมนตรี": "Prime Minister News",
        "ข่าวรองนายกรัฐมนตรี  รัฐมนตรีประจำสำนักนายกรัฐมนตรี": "Deputy Prime Minister News",
        "คำกล่าวสุนทรพจน์": "Statements",
        "ด้านกฎหมายฯ": "Legal Affairs",
        "ด้านการศึกษาฯ": "Education Affairs",
        "ด้านความมั่นคง": "Security Affairs",
        "ด้านวัฒนธรรมท่องเที่ยวฯ": "Cultural and Tourism Affairs",
        "ด้านสังคม": "Social Affairs",
        "ด้านเศรษฐกิจ": "Economic Affairs",
        "ทัน LINE ไทยคู่ฟ้า": "Official Thai Government LINE Account",
        "รายการวิทยุไทยคู่ฟ้า": "Official Thai Government Radio Program",
        "สรุปข่าวการประชุม ครม": "Cabinet Meeting Synopsis",
        "อื่นๆ": "Others",
        "ชี้แจงประเด็นสำคัญ": "Others",
        "ข่าวทำเนียบรัฐบาล": "Government House News",
        "ภารกิจ นายเศรษฐา ทวีสิน  นายกรัฐมนตรี": "Prime Minister News",
        "ข่าวรอง นรม  รมตนร": "Deputy Prime Minister News",
        "พักหนี้เกษตรกร": "Economic Affairs",
        "สร้างคุณภาพชีวิต": "Social Affairs",
        "แก้ปัญหาหนี้สิน": "Economic Affairs",
        "รัฐบาลดิจิทัล": "Government House News",
        "การดูแลสิ่งแวดล้อม": "Social Affairs",
        "การปราบปรามยาเสพติด": "Legal Affairs",
        "การผลักดัน Soft Power": "Economic Affairs",
        "การพัฒนากองทัพ": "Security Affairs",
        "การพัฒนาโครงสร้างพื้นฐาน": "Social Affairs",
        "การแก้ไขปัญหาหนี้สินภาคประชาชน": "Economic Affairs",
        "ความสัมพันธ์และความร่วมมือระหว่างประเทศ(ทวิภาคี)": "Government House News",
        "ความสัมพันธ์และความร่วมมือระหว่างประเทศ(พหุภาคี)": "Government House News",
        "นโยบาย Digital Wallet": "Economic Affairs",
        "บทบาทประเทศไทยบนเวทีโลก": "Government House News",
        "ปฏิรูปการศึกษาและสร้างสังคมแห่งการเรียนรู้ตลอดชีวิต": "Education Affairs",
        "ยกระดับคุณภาพชีวิต": "Social Affairs",
        "แก้ไขปัญหาความเห็นต่างรัฐธรรมนูญ พศ 2560": "Government House News",
        "การประคองภาระหนี้สินและต้นทุนทางการเงินภาคธุรกิจ SMEs": "Economic Affairs",
        "ภาคการค้าการลงทุน": "Economic Affairs",
        "การแก้ไขปัญหาหนี้สินภาคการเกษตร": "Economic Affairs",
        "ลดภาระค่าใช้จ่ายประชาชนด้านพลังงานและระบบสาธารณูปโภค": "Economic Affairs",
        "สร้างรายได้จากผืนดินและส่งเสริมสิ่งแวดล้อม": "Economic Affairs",
        "ภาคการท่องเที่ยว": "Cultural and Tourism Affairs",
        "สิทธิที่ทำกิน": "Legal Affairs",
    }


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
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="month", y="avg_score")
    plt.title("Average Sentiment Score by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(range(len(all_months)), all_months, rotation=45)
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
                        topic_dict[data["topic_model"]]["frequency"] += 1
                        topic_dict[data["topic_model"]]["avg_score"] += data["avg_score"]
    for topic, detail in topic_dict.items():
        detail["avg_score"] /= detail["frequency"]

    return topic_dict


def plot_frequency_and_avg_score_by_topic():
    topic_dict = get_frequency_and_avg_score_by_topic()
    topics = list(topic_dict.keys())
    for topic in topics:
        print(topic, topic_dict[topic]["frequency"])
    # df = pd.DataFrame(
    #     {
    #         "topic": topics,
    #         "frequency": [topic_dict[topic]["frequency"] for topic in topics],
    #         "avg_score": [topic_dict[topic]["avg_score"] for topic in topics],
    #     }
    # )
    # plt.figure(figsize=(10, 6))
    # sns.barplot(data=df, x="topic", y="frequency", order=df.sort_values("frequency", ascending=False).topic)
    # plt.title("Frequency by Topic")
    # plt.xlabel("Topic")
    # plt.ylabel("Frequency")
    # plt.xticks(range(len(topics)), topics, rotation=45)
    # plt.savefig("analysis/frequency_by_topic.png")
    # plt.show()


if __name__ == "__main__":
    # get_sentiment_score_distribution()
    # plot_sentiment_score_by_month()
    plot_frequency_and_avg_score_by_topic()
