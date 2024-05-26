import os
import json
from typing import List

import multiprocessing
# from LM.sentiment_build import tokenizer, model
# from LM.sentiment_pipeline import pipe as sentiment_model
from LLM.typhoon_7b import client, do_tokens_exceed_limit, typhoon_sentiment_analysis, typhoon_topic_modeling
from multiprocessing import Pool


# def split_tokens(content: List[str], tokenizer=tokenizer, max_length: int = 512):
#     """
#     Split the content's tokens from the tokenizer to the last occurrence of <_> before the max_length.
#     """
#     split_token_list = []
#     len_split_token_list = []
#     tokenizer_list = tokenizer("".join(content))["input_ids"]
#     len_tokenizer_list = len(tokenizer_list)
#     stride = int(0.97 * max_length)
#     i = 0
#     while i <= len_tokenizer_list:
#         tmp_list = tokenizer_list[i : i + stride]
#         split_token_list.append(tokenizer.decode(tmp_list))
#         len_split_token_list.append(len(tmp_list))
#         i += stride

#     return split_token_list, len_split_token_list


def do_inference(year: str, month: str, day: str):
    """
    Get the sentiment from the content for a given year, month, and day.
    """
    open_path = os.path.join("text_jsonl", year, month)
    save_path = os.path.join("sentiment_jsonl", year, month)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    with open(save_path + "/" + day + ".jsonl", "w", encoding="utf8") as outfile:
        with open(open_path + "/" + day + ".jsonl", "r", encoding="utf8") as infile:
            for line in infile:
                data = json.loads(line)
                content = list(map(lambda s: s.replace('\n', ''), data["content"]))
                # split_token_list, len_split_token_list = split_tokens(content)
                # sentiment = sentiment_model(split_token_list)
                # for idx, s in enumerate(sentiment):
                #     s["len_token"] = len_split_token_list[idx]
                # avg_score = sum(
                #     [(1 - s["score"] if s["label"] == "LABEL_0" else s["score"]) * s["len_token"] for s in sentiment]
                # ) / sum(len_split_token_list)
                print(data["title"])
                str_content = "".join(content)
                if do_tokens_exceed_limit(str_content):
                    print(f"Tokens exceed limit for {data['title']}")
                    continue
                sentiment_score = typhoon_sentiment_analysis(str_content)

                # topic_from_llm = typhoon_topic_modeling("".join(content))

                ret = {
                    "date": f"{year}-{month}-{day}",
                    "topic_from_title": data["topic_from_title"],
                    "title": data["title"],
                    "content": data["content"],
                    "sentiment": "positive" if float(sentiment_score) > 0.5 else "negative",
                    "sentiment_score": sentiment_score,
                    # "topic_from_llm": topic_from_llm,
                }

                jout = json.dumps(ret, ensure_ascii=False) + "\n"
                outfile.write(jout)

do_inference('2023', '04', '04')
########################
# !!!TODO!!!: multiprocessing problem as the tokenizers got forked before (to be fixed)
########################

# def get_txt_all_dates_parallel(n_processes: int = 4):
#     """
#     Get the lines from the txt files for all dates in parallel using multiprocessing.
#     """
#     all_dates = []
#     for year in os.listdir('text_jsonl'):
#         for month in os.listdir(os.path.join('text_jsonl', year)):
#             for day in os.listdir(os.path.join('data', year, month)):
#                 all_dates.append((year, month, day[:2]))

#     with Pool(processes=n_processes) as pool:
#         pool.starmap(do_inference, all_dates)

# if __name__ == "__main__":
#   get_txt_all_dates_parallel(n_processes=2)
