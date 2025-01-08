import os
import pandas as pd
import jsonlines
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from tqdm import tqdm
from enum import Enum
from together import Together
import json
from random import sample
import jsonlines as jsonl
from openai import OpenAI
import typing_extensions as typing
import google.generativeai as genai
import vertexai
from openai import OpenAI
from pathlib import Path
load_dotenv('/Users/guida/llm_argument_tasks/whats-your-arg/.env')

api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI()

api_key = os.environ.get('TOGETHER_API_KEY')
client = Together(api_key=api_key)

PROJECT_ID = os.environ.get('GEMINI_PROJECT_ID')
LOCATION = "us-central1"


class ModelType(Enum):
    GEMINI = "gemini"
    GPT4 = "gpt4o"
    GPT4_MINI = "gpt4o-mini"
    LLAMA = "llama"

class TaskType(Enum):
    TASK2_BINARY = "task2_binary"
    TASK2_FULL = "task2_full"

class RelationClassificationGemini(typing.TypedDict):
    id: str 
    label: int

class RelationClassification(BaseModel):
    id: str = Field(description="The ID of the comment being analyzed")
    label: int = Field(description="The label for argument relation")

class ModelConfig:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.initialize_client()

    def initialize_client(self):
        if self.model_type == ModelType.GEMINI:
            genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
            self.client = genai.GenerativeModel("gemini-1.5-flash")
        elif self.model_type in [ModelType.GPT4, ModelType.GPT4_MINI]:
            self.client = OpenAI()
        elif self.model_type == ModelType.LLAMA:
            self.client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

def create_prompt(self, id: str, topic: str, argument: str, samples: str, task_type: TaskType) -> str:
    if task_type == TaskType.TASK2_BINARY:
        return f"""
            Task: Binary Classification of Arguments about {topic}
            Target Argument: {argument}
            Role: You are an expert in argument analysis and logical reasoning, specializing in identifying rhetorical patterns in social discourse.
            Step-by-Step Instructions:
            1. Read the input text thoroughly
            2. Evaluate the text's relationship to the target argument, examining:
            - Direct support or opposition
            - Implicit agreement or disagreement
            3. Make a binary classification decision
            4. Format the output according to specifications
            Classification Rules:
            - Label = 5: Comment supports/agrees with argument
            - Label = 1: Comment attacks/disagrees with argument
            Critical Requirements:
            - Use ONLY specified labels (1 or 5)
            - Do NOT quote or repeat input texts
            - Return VALID JSON only
            {f'Examples:\n{samples}' if samples else ''}
            Output Schema:
            {{
                "id": "{id}",
                "label": label_value  # must be 1 or 5 without quotes
            }}
        """
    elif task_type == TaskType.TASK2_FULL:
        return f"""
            Task: Classification of Arguments about {topic}
            Target Argument: {argument}
            Role: You are an expert in argument analysis and logical reasoning, specializing in identifying rhetorical patterns in social discourse.
            Step-by-Step Instructions:
            1. Read the input text thoroughly
            2. Evaluate the text's relationship to the target argument, examining:
            - Direct support or opposition
            - Implicit agreement or disagreement
            3. Make a classification decision
            4. Format the output according to specifications
            Classification Rules:
            - Label = 5: Comment supports/agrees with argument explicitly
            - Label = 4: Comment supports/agrees with argument implicitly/indirectly
            - Label = 2: Comment attacks/disagrees with argument implicitly/indirectly
            - Label = 1: Comment attacks/disagrees with argument explicitly
            Critical Requirements:
            - Use ONLY specified labels (1, 2, 4 or 5)
            - Do NOT quote or repeat input texts
            - Return VALID JSON only
            {f'Examples:\n{samples}' if samples else ''}
            Output Schema:
            {{
                "id": "{id}",
                "label": label_value  # must be 1, 2, 4 or 5 without quotes
            }}
        """

def classify_text(self, id: str, comment_text: str, topic: str, argument: str, samples: str, task_type: TaskType) -> dict:
    prompt = self.create_prompt(id, topic, argument, samples, task_type)
    
    try:
        if self.model_type == ModelType.GEMINI:
            full_prompt = f"{prompt}\nComment: {comment_text}"
            response = self.client.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=RelationClassificationGemini,
                        temperature=0,
                        top_p=1,
                    ),
                    safety_settings={
                        "HARM_CATEGORY_HARASSMENT": "block_none",
                        "HARM_CATEGORY_HATE_SPEECH": "block_none",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none"
                    }
                )

            return json.loads(response.text)
        
        elif self.model_type in [ModelType.GPT4, ModelType.GPT4_MINI]:
            model_name = "gpt-4o" if self.model_type == ModelType.GPT4 else "gpt-4o-mini"
            completion = self.client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": comment_text}
                ],
                response_format=RelationClassification,
                temperature=0
            )
            return json.loads(completion.choices[0].message.content)
        
        elif self.model_type == ModelType.LLAMA:
            extract = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": comment_text}
                ],
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                temperature=0,
                top_k=1,
                top_p=1,
                response_format={
                    "type": "json_object",
                    "schema": RelationClassification.model_json_schema(),
                }
            )
            return json.loads(extract.choices[0].message.content)
    
    except Exception as e:
        print(f"Error in classification: {e}")
        return {"id": id, "label": 3}

def prep_fewshot_samples_task2(samples_file: str, task_type: TaskType, n: int) -> str:
    df = pd.read_csv(samples_file)
    ids = df['id'].to_list()
    sampled = sample(ids, n)
    df = df[df['id'].isin(sampled)]
    
    output = ""
    for i, row in df.iterrows():
        comment = row['comment_text']
        argument = row['argument_text']
        label = row['label']
        
        if task_type == TaskType.TASK2_BINARY:
            if label == 2: label = 1
            if label == 4: label = 5
            
        output += f"Comment: {comment}\nArgument: {argument}\nLabel: {label}\n\n"
    
    return output

class DataProcessor:
    def __init__(self, model_config: ModelConfig, output_dir: Path):
        self.model_config = model_config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_dataset(
        self,
        df: pd.DataFrame,
        topic: str,
        task_type: TaskType,
        shot_type: int,
        split: Optional[int] = None,
        samples: str = ""
    ):
        output_filename = self.get_output_filename(topic, task_type, shot_type, split)
        
        with jsonl.open(self.output_dir / output_filename, mode='w') as writer:
            for _, row in tqdm(df.iterrows(), total=len(df)):
                try:
                    comment_id = row['id']
                    comment_text = row['comment_text']
                    argument = row['argument_text']

                    classification = self.model_config.classify_text(
                        id=comment_id,
                        comment_text=comment_text,
                        topic=topic,
                        argument=argument,
                        samples=samples,
                        task_type=task_type
                    )
                    
                    writer.write({"id": comment_id, "label": classification["label"]})
                    print(classification)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    writer.write({"id": comment_id if 'comment_id' in locals() else "error", "label": 3})

    def get_output_filename(self, topic: str, task_type: TaskType, shot_type: int, split: Optional[int]) -> str:
        model_name = self.model_config.model_type.value
        split_suffix = f"_split_{split}" if split is not None else ""
        return f"{task_type.value}_{model_name}_comarg_{topic}_{shot_type}shot{split_suffix}.jsonl"

def main():
    output_dir = Path("task2_outputs")
    models = [ModelType.GEMINI, ModelType.GPT4, ModelType.GPT4_MINI, ModelType.LLAMA]
    task_types = [TaskType.TASK2_BINARY, TaskType.TASK2_FULL]
    shot_types = [0, 1, 5]
    splits = list(range(1, 6))

    comarg_files = ["GM_all_arguments_main", "UGIP_all_arguments_main"]

    for model_type in models:
        model_config = ModelConfig(model_type)
        processor = DataProcessor(model_config, output_dir)

        for task_type in task_types:
            for file_name in comarg_files:
                df = pd.read_csv(f"data/{file_name}.csv")
                processor.process_dataset(df, file_name, task_type, 0)

                for shot in [1, 5]:
                    for split in splits:
                        df = pd.read_csv(f"data/k-shots/{file_name}_{shot}shot_split_{split}.csv")
                        samples = prep_fewshot_samples_task2(f"data/k-shots/{file_name}_{shot}shot_split_{split}.csv", task_type, shot)
                        processor.process_dataset(df, file_name, task_type, shot, split, samples)

if __name__ == "__main__":
    main()