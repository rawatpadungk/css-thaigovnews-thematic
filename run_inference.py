import os
import json
from typing import List
# from LM.sentiment_build import tokenizer, model
from LM.sentiment_pipeline import pipe as sentiment_model
 
tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512}

def get_split_max_length(c: str, ret: List = []):
    """
    Get the split max length from the content.
    """
    if len(c) <= 512:
        ret.append(c)
        return ret
    else:
        last_space_idx_bf_512 = c[:512].rfind('<_>')
        content_first = c[:last_space_idx_bf_512]
        content_second = c[last_space_idx_bf_512:]
        ret.append(content_first)
        return get_split_max_length(content_second, ret)


def get_sentiment(year: int, month: int, day: int):
    """
    Get the sentiment from the content for a given year, month, and day.
    """
    open_path = os.path.join('text_jsonl', str(year), str(month).zfill(2))
    save_path = os.path.join('sentiment_jsonl', str(year), str(month).zfill(2))
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
 
    with open(save_path + '/' + str(day).zfill(2) + '.jsonl', 'w', encoding='utf8') as outfile:
        with open(open_path + '/' + str(day).zfill(2) + '.jsonl', 'r', encoding='utf8') as infile:
            for line in infile:
                data = json.loads(line)
                content = data['content']
                new_content = []
                for c in content:
                    if len(c) <= 512:
                        new_content.append(c)
                    else:
                        new_content.extend(get_split_max_length(c))
                sentiment = sentiment_model(new_content, **tokenizer_kwargs)
                for idx, s in enumerate(sentiment):
                    s['len_str'] = len(new_content[idx])
                ret = {
                    'date': f'{year}-{month}-{day}',
                    'topic': data['topic'],
                    'content': data['content'],
                    'sentiment': sentiment,
                }
                jout = json.dumps(ret, ensure_ascii=False) + '\n'
                outfile.write(jout)
                

def get_sentiment_all_dates():
    """
    Get the sentiment from the content for all dates.
    """
    for year in os.listdir('text_jsonl'):
        for month in os.listdir(os.path.join('text_jsonl', year)):
            for day in os.listdir(os.path.join('text_jsonl', year, month)):
                get_sentiment(int(year), int(month), int(day))
    # get_sentiment(2023, 4, 4)

get_sentiment_all_dates()