import os
import json
from typing import List

import multiprocessing
from LM.sentiment_build import tokenizer, model
from LM.sentiment_pipeline import pipe as sentiment_model


def split_tokens(content: List[str], tokenizer=tokenizer, max_length: int = 512):
    """
    Split the content's tokens from the tokenizer to the last occurrence of <_> before the max_length.
    """
    split_token_list = []
    len_split_token_list = []
    tokenizer_list = tokenizer("".join(content))["input_ids"]
    len_tokenizer_list = len(tokenizer_list)
    stride = int(0.97 * max_length)
    i = 0
    while i <= len_tokenizer_list:
        tmp_list = tokenizer_list[i : i + stride]
        split_token_list.append(tokenizer.decode(tmp_list))
        len_split_token_list.append(len(tmp_list))
        i += stride

    return split_token_list, len_split_token_list


def get_sentiment(year: int, month: int, day: int):
    """
    Get the sentiment from the content for a given year, month, and day.
    """
    open_path = os.path.join("text_jsonl", str(year), str(month).zfill(2))
    save_path = os.path.join("sentiment_jsonl", str(year), str(month).zfill(2))
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    with open(save_path + "/" + str(day).zfill(2) + ".jsonl", "w", encoding="utf8") as outfile:
        with open(open_path + "/" + str(day).zfill(2) + ".jsonl", "r", encoding="utf8") as infile:
            for line in infile:
                data = json.loads(line)
                content = data["content"]
                split_token_list, len_split_token_list = split_tokens(content)
                sentiment = sentiment_model(split_token_list)
                for idx, s in enumerate(sentiment):
                    s["len_token"] = len_split_token_list[idx]
                avg_score = sum(
                    [(1 - s["score"] if s["label"] == "LABEL_0" else s["score"]) * s["len_token"] for s in sentiment]
                ) / sum(len_split_token_list)

                ret = {
                    "date": f"{year}-{month}-{day}",
                    "topic": data["topic"],
                    "content": data["content"],
                    "sentiment": sentiment,
                    "avg_score": avg_score,
                    "sentiment_type": "positive" if avg_score > 0.5 else "negative",
                }

                jout = json.dumps(ret, ensure_ascii=False) + "\n"
                outfile.write(jout)


def get_all_dates():
    """
    Get the sentiment from the content for all dates.
    """
    all_dates = []
    for year in os.listdir("text_jsonl"):
        for month in os.listdir(os.path.join("text_jsonl", year)):
            for day in os.listdir(os.path.join("text_jsonl", year, month)):
                all_dates.append((int(year), int(month), int(day[:2])))
    return all_dates


if __name__ == "__main__":
    all_dates = get_all_dates()
    with multiprocessing.Pool(3) as p:
        p.starmap(get_sentiment, all_dates)
