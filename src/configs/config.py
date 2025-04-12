import json


class Settings:
    with open('src/configs/apikey.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        OPENAIAPIKEY = data.get('openAI_apikey', '')
        DISCORDTOKEN = data.get('discord', {}).get('token', '')
        DISCORDPREFIX = data.get('discord', {}).get('prefix', '!')
