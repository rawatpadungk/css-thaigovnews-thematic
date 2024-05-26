from openai import OpenAI
import config
from transformers import AutoTokenizer

client = OpenAI(
    api_key=config.OPENTYPHOON_API_KEY,
    base_url="https://api.opentyphoon.ai/v1",
)

tokenizer = AutoTokenizer.from_pretrained("scb10x/typhoon-7b")

def do_tokens_exceed_limit(text, max_length=4096):
    """
    Check if the tokens exceed the max_length.
    """
    tokens = tokenizer(text)["input_ids"]
    return len(tokens) > max_length

def typhoon_sentiment_analysis(transcription):
    response = client.chat.completions.create(
        model="typhoon-instruct",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "ทำการวิเคราะห์ความรู้สึกของข่าวสารรัฐบาลไทยและตอบแค่คะแนนระหว่าง 0 ถึง 1 โดยที่ 0 แสดงถึงความไม่พอใจอย่างแรง และ 1 แสดงถึงความพึงพอใจอย่างแรง"
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content

def typhoon_topic_modeling(transcription):
    response = client.chat.completions.create(
        model="typhoon-instruct",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "ทำการจำแนกหัวข้อหลักจากข่าวสารรัฐบาลไทยและตอบแค่หัวข้อที่แสดงให้เห็นภาพชัดเจนที่สุด"
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content