import os
import glob
import json
from typing import List
import re
from multiprocessing import Pool


def get_lines_from_txt(year: str, month: str, day: str):
    """
    Get the lines from the txt file for a given year, month, and day.
    """
    open_path = os.path.join('data', year, month, day)
    save_path = os.path.join('text_jsonl', year, month)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    with open(save_path + '/' + day + '.jsonl', 'w', encoding='utf8') as outfile:
        for file in glob.glob(open_path + '/*.txt'):
            with open(file, 'r') as f:
                lines = f.readlines()
                lines = list(map(lambda s: s.replace('\xa0', ' '), lines))
                ret = {
                    'date': f'{year}-{month}-{day}',
                    'topic_from_title': file.split('/')[-1].split('.')[0].split('_')[0],
                    'title': lines[0].replace('\n', ''),
                    'content': get_content(lines[1:-1]),
                }
                jout = json.dumps(ret, ensure_ascii=False) + '\n'
                outfile.write(jout)


def get_content(lines: List):
    """
    Get the content from the lines.
    P.S. http addresses are still not removed.
    """
    content = []
    for line in lines:
        # rule-based filtering
        if line == '\n': continue
        elif line == 'พิมพ์' + '\n': continue
        elif re.match(r'\d{2}/\d{2}/\d{4}', line[:-1]): continue
        elif re.match(r'วัน[ก-๙]{3,8}ที่ \d{1,2} [ก-๙]{6,10} \d{4}', line[:-1]): continue
        elif len(''.join(set(line[:-1]))) == 1: continue
        else:
            # content.append(line.replace('\n', ''))
            content.append(line)
    return list(dict.fromkeys(content))


def get_txt_all_dates_parallel(n_processes: int = 4):
    """
    Get the lines from the txt files for all dates in parallel using multiprocessing.
    """
    all_dates = []
    for year in os.listdir('data'):
        for month in os.listdir(os.path.join('data', year)):
            for day in os.listdir(os.path.join('data', year, month)):
                all_dates.append((year, month, day[:2]))

    with Pool(processes=n_processes) as pool:
        pool.starmap(get_lines_from_txt, all_dates)

if __name__ == "__main__":
  get_txt_all_dates_parallel(n_processes=2)