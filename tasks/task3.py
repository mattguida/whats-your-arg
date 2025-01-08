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

topic_label_to_argument = {
    "abortion": {
        "p-right": "Abortion is a woman’s right.",
        "p-rape": "Rape victims need it to be legal.",
        "p-not_human": "A fetus is not a human yet, so it's okay to abort.",
        "p-mother_danger": "Abortion should be allowed when a mother's life is in danger.",
        "p-baby_ill_treatment": "Unwanted babies are ill-treated by parents and/or not always adopted.",
        "p-birth_ctrl": "Birth control fails at times and abortion is one way to deal with it.",
        "p-not_murder": "Abortion is not murder.",
        "p-sick_mom": "Mother is not healthy/financially solvent.",
        "p-other": "Others",
        "c-adopt": "Put baby up for adoption.",
        "c-kill": "Abortion kills a life.",
        "c-baby_right": "An unborn baby is a human and has the right to live.",
        "c-sex": "Be willing to have the baby if you have sex.",
        "c-bad_4_mom": "Abortion is harmful for women.",
        "c-other": "Others"
    },
    "gayRights": {
        "p-normal": "Gay marriage is like any other marriage.",
        "p-right_denied": "Gay people should have the same rights as straight people.",
        "p-no_threat_for_child": "Gay parents can adopt and ensure a happy life for a baby.",
        "p-born": "People are born gay.",
        "p-religion": "Religion should not be used against gay rights.",
        "p-Other": "Others",
        "c-religion": "Religion does not permit gay marriages.",
        "c-abnormal": "Gay marriages are not normal/against nature.",
        "c-threat_to_child": "Gay parents cannot raise kids properly.",
        "c-gay_problems": "Gay people have problems and create social issues.",
        "c-Other": "Others"
    },
    "obama": {
        "p-economy": "Fixed the economy.",
        "p-War": "Ending the wars.",
        "p-republicans": "Better than the republican candidates.",
        "p-decision_policies": "Makes good decisions/policies.",
        "p-quality": "Has qualities of a good leader.",
        "p-health": "Ensured better healthcare.",
        "p-foreign_policies": "Executed effective foreign policies.",
        "p-job": "Created more jobs.",
        "p-Other": "Others",
        "c-economy": "Destroyed our economy.",
        "c-War": "Wars are still on.",
        "c-job": "Unemployment rate is high.",
        "c-health": "Healthcare bill is a failure.",
        "c-decision_policies": "Poor decision-maker.",
        "c-republicans": "We have better republicans than Obama.",
        "c-quality": "Not eligible as a leader.",
        "c-foreign_policies": "Ineffective foreign policies.",
        "c-Other": "Others"
    },
    "marijuana": {
        "p-not_addictive": "Not addictive.",
        "p-medicine": "Used as a medicine for its positive effects.",
        "p-legal": "Legalized marijuana can be controlled and regulated by the government.",
        "p-right": "Prohibition violates human rights.",
        "p-no_damage": "Does not cause any damage to our bodies.",
        "p-Other": "Others",
        "c-health": "Damages our bodies.",
        "c-mind": "Responsible for brain damage.",
        "c-illegal": "If legalized, people will use marijuana and other drugs more.",
        "c-crime": "Causes crime.",
        "c-addiction": "Highly addictive.",
        "c-Other": "Others"
    }
}
def prep_fewshot_samples(samples_file, topic, n):
    df = pd.read_csv(samples_file)

    if n != 5:
        ids = df['uid'].to_list()
        sampled = sample(ids, n)
        df = df[df['uid'].isin(sampled)]
    
    output = ''
    for i, row in df.iterrows():
        comment = row['text']
        output = f"{output}\n Comment: {comment}\n"
        argument_type = row['label']
        argument = topic_label_to_argument[topic][argument_type]
        output = f"{output} Argument {i}: {argument}\n"
        span = row['text']
        output = f"{output} Span: {span}\n\n"
    return output

class ModelType(Enum):
    GEMINI = "gemini"
    GPT4 = "gpt4o"
    GPT4_MINI = "gpt4o-mini"
    LLAMA = "llama"

class DatasetType(Enum):
    YRU = "yru"

class ArgumentSpanGemini(typing.TypedDict):
    id: str 
    span: str 

class ArgumentSpan(BaseModel):
    id: str = Field(description="The ID of the comment being analyzed")    
    span: str = Field(description="The span of text in the comment that makes use of the argument") 

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

    def classify_text(self, id: str, comment: str, topic: str, argument: str, samples: str = "") -> dict:
        prompt = self.create_prompt(id, topic, argument, samples)
        
        try:
            if self.model_type == ModelType.GEMINI:
                full_prompt = f"{prompt}\nComment: {comment}"
                response = self.client.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=ArgumentSpanGemini,
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
                print(prompt)
                print(json.loads(response.text))
                return json.loads(response.text)
            
            elif self.model_type in [ModelType.GPT4, ModelType.GPT4_MINI]:
                model_name = "gpt-4o" if self.model_type == ModelType.GPT4 else "gpt-4o-mini"
                completion = self.client.beta.chat.completions.parse(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": comment}
                    ],
                    response_format=ArgumentSpan,
                    temperature=0
                )

                return json.loads(completion.choices[0].message.content)
            
            elif self.model_type == ModelType.LLAMA:
                extract = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": comment}
                    ],
                    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                    temperature=0,
                    top_k=1,
                    top_p=1,
                    response_format={
                        "type": "json_object",
                        "schema": ArgumentSpan.model_json_schema(),
                    }
                )
                return json.loads(extract.choices[0].message.content)
        
        except Exception as e:
            print(f"Error in classification: {e}")
            return {"id": id, "label": 0}

    def create_prompt(self, id: str, topic: str, argument: str, samples: str) -> str:
        return f"""
                Task: Text Span Identification for Arguments about {topic}
                Target Argument: {argument}
                Role: You are an expert in argument analysis and logical reasoning, specializing in identifying rhetorical patterns in social discourse.

                Step-by-Step Instructions:
                1. Read the input text carefully
                2. Locate exact text spans that:
                - Directly reference the target argument
                - Express the same idea as the argument
                3. Extract the precise text span
                4. Format the output according to specifications

                Critical Requirements:
                - Extract EXACT text only (no paraphrasing)
                - Include COMPLETE relevant phrases
                - Use MINIMUM necessary context
                - Maintain ORIGINAL formatting
                - Return VALID JSON only

                Output Schema:
                {{
                    "id": "{id}",
                    "span": "exact_text_from_comment"  # must be verbatim quote
                }}
            
            {f'Some examples:\n{samples}' if samples else ''}
        """

class DataProcessor:
    def __init__(self, model_config: ModelConfig, output_dir: Path):
        self.model_config = model_config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_dataset(
        self,
        df: pd.DataFrame,
        topic: str,
        shot_type: int,
        split: Optional[int] = None,
        samples: str = ""
    ):
        output_filename = self.get_output_filename(topic, shot_type, split)
        
        with jsonl.open(self.output_dir / output_filename, mode='w') as writer:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {topic}"):
                try:
                    comment_id = row['uid']
                    comment = row['text']
                    argument = topic_label_to_argument[topic][row['label']]

                    classification = self.model_config.classify_text(
                        id=comment_id,
                        comment=comment,
                        topic=topic,
                        argument=argument,
                        samples=samples
                    )
                    print(id, comment, topic, argument)
                    writer.write({
                        "id": comment_id, 
                        "span": classification["span"]
                    })
                    
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    writer.write({
                        "id": comment_id if 'comment_id' in locals() else f"error_{idx}", 
                        "span": ""
                    })

    def get_output_filename(self, topic: str, shot_type: int, split: Optional[int]) -> str:
        model_name = self.model_config.model_type.value
        split_suffix = f"_split_{split}" if split is not None else ""
        return f"task3_{model_name}_YRU_{topic}_{shot_type}shot{split_suffix}.jsonl"

def load_dataset(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def prepare_samples(file_path: str, topic: str, n_shots: int) -> str:
        return prep_fewshot_samples(file_path, topic)

# Main execution
def main():
    output_dir = Path("task3_outputs")
    models = [ModelType.GEMINI, ModelType.GPT4, ModelType.GPT4_MINI, ModelType.LLAMA]
    shot_types = [0, 1, 5]
    splits = list(range(1, 6))  

    for model_type in models:
        model_config = ModelConfig(model_type)
        processor = DataProcessor(model_config, output_dir)
        
        # Process YRU datasets
        yru_topics = ["abortion", "gayRights", "marijuana", "obama"]
        for topic in yru_topics:
            # 0-shot
            df = load_dataset(f"data/yru_{topic}_with_negatives_main.csv")
            processor.process_dataset(df, topic, 0)

            # k-shot
            for shot in [1, 5]:
                for split in splits:
                    df = load_dataset(f"data/k-shots/yru_{topic}_with_negatives_main_{shot}shot_split_{split}.csv")
                    samples = prepare_samples(f"data/k-shots/yru_{topic}_with_negatives_main_{shot}shot_split_{split}.csv", topic, shot)
                    processor.process_dataset(df, topic, shot, split, samples)

if __name__ == "__main__":
    main()