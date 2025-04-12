import json

class Settings:
    def __init__(self):
        pass
    def _get_openapi_key(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['openAI_apikey']
    OPENAPIKEY = _get_openapi_key('src/core/apikey.json')