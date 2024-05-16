import os
import glob
import json
from typing import List
from LM.preprocess import process_transformers


def get_lines_from_txt(year: int, month: int, day: int):
    """
    Get the lines from the txt file for a given year, month, and day.
    """
    open_path = os.path.join('data', str(year), str(month).zfill(2), str(day).zfill(2))
    save_path = os.path.join('text_jsonl', str(year), str(month).zfill(2))
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    with open(save_path + '/' + str(day).zfill(2) + '.jsonl', 'w', encoding='utf8') as outfile:
        for file in glob.glob(open_path + '/*.txt'):
            with open(file, 'r') as f:
                lines = f.readlines()
                lines = list(map(lambda s: s.replace('\xa0', ' '), lines))
                lines = [process_transformers(line) for line in lines]
                ret = {
                    'date': f'{year}-{month}-{day}',
                    'topic': lines[0],
                    'content': get_content(lines),
                }
                jout = json.dumps(ret, ensure_ascii=False) + '\n'
                outfile.write(jout)


def get_content(lines: List):
    """
    Get the content from the lines.
    P.S. This is not complete. Need to handle the irrelevant lines.
    """
    content = []
    for line in lines[1:-1]:
        if line == '': continue
        content.append(line)
    return content


def get_txt_all_dates():
    """
    Get all the dates that we have data for.
    """
    for year in os.listdir('data'):
        for month in os.listdir(os.path.join('data', year)):
            for day in os.listdir(os.path.join('data', year, month)):
                get_lines_from_txt(int(year), int(month), int(day))
#     get_lines_from_txt(2023, 1, 1)

# get_txt_all_dates()