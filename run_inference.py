import os
import json
from typing import List

from LM.sentiment_build import tokenizer, model
from LM.sentiment_pipeline import pipe as sentiment_model

# tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}


def split_tokens(content: List[str], tokenizer=tokenizer, max_length: int = 512):
    """
    Split the content's tokens from the tokenizer to the max length.
    """
    ret = []
    token_list = tokenizer("".join(content))["input_ids"]
    len_token_list = len(token_list)
    stride = max_length - 5
    for i in range(0, len_token_list, stride):
        ret.append(tokenizer.decode(token_list[i : i + stride]))
    return ret


def get_split_max_length(c: str, ret: List = []):
    """
    Get the split max length from the content.
    """
    if len(c) <= 512:
        ret.append(c)
        return ret
    else:
        last_space_idx_bf_512 = c[:512].rfind("<_>")
        content_first = c[:last_space_idx_bf_512]
        content_second = c[last_space_idx_bf_512:]
        ret.append(content_first)
        return get_split_max_length(content_second, ret)


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
                new_content = split_tokens(content)
                sentiment = sentiment_model(new_content)
                for idx, s in enumerate(sentiment):
                    s["len_str"] = len(new_content[idx])
                avg_score = sum([s["score"] for s in sentiment]) / len(sentiment)
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


def get_sentiment_all_dates():
    """
    Get the sentiment from the content for all dates.
    """
    for year in os.listdir("text_jsonl"):
        for month in os.listdir(os.path.join("text_jsonl", year)):
            for day in os.listdir(os.path.join("text_jsonl", year, month)):
                # try:
                #     get_sentiment(int(year), int(month), int(day[:2]))
                # except:
                #     print(f"Error: {year}-{month}-{day[:2]}")
                get_sentiment(int(year), int(month), int(day[:2]))


get_sentiment_all_dates()
