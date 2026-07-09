from datasets import load_dataset
from system_prompts import get_sys_prompt, get_sys_prompt_formatted
import numpy as np 

def process_dataset(dataset, script_args):
    num_candidates = script_args.num_candidates if "num_candidates" in script_args.__dict__.keys() else 1
    sys_prompt = get_sys_prompt_formatted(script_args.sys_prompt_name, num_candidates)


    if script_args.task_spec == "gen":
        dataset_name = str(getattr(script_args, "dataset_name", "")).lower()
        format_pattern = str(getattr(script_args, "format_pattern", "")).lower()
        if "musique" in dataset_name or format_pattern == "musique_multi_answer":
            dataset = make_musique_generation_dataset(dataset, sys_prompt)
        else:
            dataset = make_generation_dataset(dataset, sys_prompt)

    return dataset

def _format_musique_paragraphs(paragraphs):
    formatted = []
    for paragraph in paragraphs or []:
        idx = paragraph.get("idx", len(formatted)) if isinstance(paragraph, dict) else len(formatted)
        title = paragraph.get("title", "") if isinstance(paragraph, dict) else ""
        text = paragraph.get("paragraph_text", "") if isinstance(paragraph, dict) else str(paragraph)
        title_part = f" (Title: {title})" if title else ""
        formatted.append(f"[{idx}]{title_part} {text}".strip())
    return "\n".join(formatted)

def make_musique_generation_dataset(dataset, sys_prompt):
    def make_musique_conversation(example):
        paragraphs_text = _format_musique_paragraphs(example.get("paragraphs", []))
        user_format = (
            "Read the following indexed paragraphs and answer the question.\n\n"
            f"PARAGRAPHS:\n{paragraphs_text}\n\n"
            f"QUESTION: {example['question']}\n"
        )
        result = {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_format},
            ],
        }
        for key in (
            "answer",
            "answer_aliases",
            "paragraphs",
            "question_decomposition",
            "answerable",
            "id",
        ):
            if key in example:
                result[key] = example[key]
        return result

    if hasattr(dataset, "keys"):
        from datasets import DatasetDict
        mapped_splits = {}
        for split_name in dataset.keys():
            mapped_splits[split_name] = dataset[split_name].map(make_musique_conversation)
        return DatasetDict(mapped_splits)
    return dataset.map(make_musique_conversation)

def make_generation_dataset(dataset,sys_prompt):
    def make_generation_conversation(example):
        if 'question' in example.keys():
            user_format = (
                f"\n\nPROBLEM: {example['question']}\n\n"
                )
        elif 'problem' in example.keys():
            user_format = (
                    f"\n\nPROBLEM: {example['problem']}\n\n"
                    )
        else:
            user_format = (
                f"\n\nWRITING PROMPT: {example['prompt']}\n\n"
                )
        result = {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_format},
            ],
        }
        # Preserve answer/answers field if it exists
        if "answer" in example:
            result["answer"] = example["answer"]
        if "answers" in example:
            result["answers"] = example["answers"]
        return result
    
    # For DatasetDict, we need to map over each split individually
    # The mapping function preserves answer/answers fields, so columns are preserved by default
    if hasattr(dataset, 'keys'):  # It's a DatasetDict
        from datasets import DatasetDict
        mapped_splits = {}
        for split_name in dataset.keys():
            mapped_splits[split_name] = dataset[split_name].map(make_generation_conversation)
        dataset = DatasetDict(mapped_splits)
    else:  # It's a single Dataset
        dataset = dataset.map(make_generation_conversation)
    return dataset



