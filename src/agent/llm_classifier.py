from __future__ import annotations
import json, os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

class LLMIntentClassifier:
    def __init__(self, taxonomy_path: str = 'src/intents/taxonomy.json'):
        self.taxonomy = json.loads(Path(taxonomy_path).read_text(encoding='utf-8'))

    def predict_one(self, text: str) -> tuple[str, float]:
        if not os.getenv('OPENAI_API_KEY'):
            from src.intents.rules import heuristic_intent
            return heuristic_intent(text)
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        labels = list(self.taxonomy.keys())
        schema = {
            'type':'object',
            'properties': {
                'intent': {'type':'string','enum':labels},
                'confidence': {'type':'number','minimum':0,'maximum':1}
            },
            'required':['intent','confidence'],
            'additionalProperties':False
        }
        prompt = {
            'taxonomy': self.taxonomy,
            'message': text,
            'instruction': 'Choose exactly one intent. Use other when evidence is insufficient.'
        }
        r=client.responses.create(
            model=os.getenv('OPENAI_CLASSIFIER_MODEL', os.getenv('OPENAI_MODEL','gpt-5.6-luna')),
            instructions='Classify customer-support messages conservatively using only the supplied taxonomy.',
            input=json.dumps(prompt,ensure_ascii=False),
            text={'format':{'type':'json_schema','name':'intent_classification','strict':True,'schema':schema}}
        )
        obj=json.loads(r.output_text)
        return obj['intent'], float(obj['confidence'])
