import time
from typing import Any
from ray.util import ActorPool
from tools.prompts import select_rxclass_prompt, RXCLASS_DRUGS_PATH, query_trials_prompt, generate_trials_prompt
from tools.ClinicalTrialGov import search_clinical_trials, count_num_trials
from pathlib import Path
import orjson
from vllm import SamplingParams
# Note: I found using StructuredOutputsParams help avoid hallucinations 
# at the cost of limiting Nemotron's ability to output its reasoning
from vllm.sampling_params import StructuredOutputsParams

# Documentation recommends temperature=1.0 and top_p=0.95 for most tasks
# https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
MAX_TOKENS = 6400
SEED = 12345
TEMPERATURE = 1
TOP_P = 0.95
RXCLASS_MAPPING = {}

for txt_file in RXCLASS_DRUGS_PATH.glob("*.txt"):
    class_type = txt_file.stem
    with txt_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, class_id = line.split("\t", 1)
            
            if name in RXCLASS_MAPPING:
                raise RuntimeError(
                    f"RxClass '{name}' exists in multiple class type files: "
                    f"[{RXCLASS_MAPPING[name]['classType']}, {class_type}]"
                )
                
            RXCLASS_MAPPING[name] = {
                "classType": class_type,
                "className": name,
                "classId": class_id,
            }

RXCLASS_NAMES = list(RXCLASS_MAPPING.keys())

RXCLASS_INDEX_TO_NAME = {
    i: name
    for i, name in enumerate(RXCLASS_NAMES)
}

def batch_chat(model: ActorPool, sampling_params: SamplingParams, prompts: list[list[dict[str, str]]], tools: list[dict[str, Any]] | None = None, batch_size: int = 16) -> list[str]:
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    
    worker_outputs = list(model.map(
        lambda actor, batch: actor.chat.remote(messages=batch, sampling_params=sampling_params, tools=tools),
        batches
    ))

    responses = []
    for worker_out in worker_outputs:
        for req_out in worker_out.get("result", []):
            responses.append(req_out.outputs[0].text)

    return responses

# Making Nemotron select an index instead of the class name helps prevent hallucinations
SELECT_RXCLASS_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": (
                "A justification (strictly 2-3 sentences) for choosing "
                "the selected index. Do not include XML, HTML, Markdown, "
                "or special tags."
            ),
        },
        "index": {
            "type": "integer",
            "minimum": 0,
            "maximum": len(RXCLASS_NAMES) - 1,
            "description": (
                "The zero-based index of the selected RxClass."
            )
        }
    },
    "required": ["reasoning", "index"],
    "additionalProperties": False,
}

def select_rxclass_task(model: ActorPool, patient_notes: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str], list[str], float]:
    prompts = [select_rxclass_prompt(patient_note) for patient_note in patient_notes]
    grouped_prompts = [orjson.dumps(p, option=orjson.OPT_INDENT_2).decode("utf-8") for p in prompts]
    structured_outputs_params = StructuredOutputsParams(json=SELECT_RXCLASS_SCHEMA)
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, seed=SEED, structured_outputs=structured_outputs_params)
    start_time = time.perf_counter()
    raw_outputs = batch_chat(model, sampling_params, prompts)
    elapsed = time.perf_counter() - start_time

    selected_classes = []
    for prompt, response in zip(prompts, raw_outputs):
        try:
            result = orjson.loads(response)
        except orjson.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON. The model likely hit a repetition loop and was truncated.\n"
                f"JSON Error: {e}\n\n"
                f"Prompt:\n{prompt}\n\n"
                f"Raw Model Output:\n{response}"
            ) from e
        index = result.get("index")
        
        if index is None:
            raise ValueError(f"""
                Index is missing\n
                Here is the prompt:\n\n
                {prompt}\n\n
                Here is the output:\n\n
                {response}
                """
            )

        class_name = RXCLASS_INDEX_TO_NAME.get(index)

        if class_name is None:
            raise ValueError(f"""
                Invalid RxClass index: {index}

                Valid range: 0-{len(RXCLASS_NAMES)-1}

                Response:

                {response}
                """
            )

        match = RXCLASS_MAPPING.get(class_name)

        if not match:
            raise FileNotFoundError(f"""
                RxClass '{class_name}' does not exist in any RxClass .txt file.
                Here is the prompt:\n\n
                {prompt}\n\n
                Here is the output:\n\n
                {response}
                """
            )

        selected_classes.append(match)

    return selected_classes, raw_outputs, grouped_prompts, elapsed

QUERY_TRIALS_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": (
                "A justification (strictly 2-3 sentences) for why these "
                "search queries were chosen. Do not include XML, HTML, "
                "Markdown, or special tags."
            ),
        },
        "queries": {
            "type": "array",
            "description": "A list of search query objects.",
            "items": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": ["string", "null"],
                        "description": """
                            This filter is a tool to help focus your search for clinical studies
                            using terms that don't fit in the other basic filters. For example,
                            you can use this filter to search for studies with a specific word in
                            the study title or study description.

                            Other examples include:
                            - The NCT number
                            - The acronym for a study
                            - A biomarker or gene name
                            - An outcome measure

                            Avoid using this filter to type in several different search terms or
                            terms that fit better in the other basic filters.

                            Note that the term field searches everything. If you search for
                            'cancer' here, you'll get trials where cancer appears in the title,
                            conditions, interventions, description, eligibility criteria, etc.

                            Use null if no free-text term is needed.
                        """
                    },
                    "condition": {
                        "type": ["string", "null"],
                        "description": """
                            Use this field to search for studies related to a condition, disease,
                            disorder, syndrome, illness, or injury (for example, breast cancer or
                            high blood pressure).

                            If searching for more than one condition, enter each separately.

                            Use null if no condition is specified.
                        """
                    },
                    "intervention": {
                        "type": ["string", "null"],
                        "description": """
                            Use this field to search for studies that use a specific drug,
                            medical device, procedure, or lifestyle change (for example,
                            radiation therapy or a low-fat diet).

                            Use null if no intervention is specified.
                        """
                    },
                },
                "required": [
                    "term",
                    "condition",
                    "intervention",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "reasoning",
        "queries",
    ],
    "additionalProperties": False,
}

def query_trials_task(model: ActorPool, patient_notes: list[dict[str, str]], drug_members: list[list[dict[str, Any]]], output_path: Path, num_queries: int = 5) -> tuple[list[list[list[dict[str, Any]]]], list[str], list[str], list[tuple[int, int]], float]:
    prompts = []
    counts = []
    
    for i, patient_note in enumerate(patient_notes):
        patient_drugs = drug_members[i]
        counts.append(len(patient_drugs))
        for drug in patient_drugs:
            prompts.append(query_trials_prompt(patient_note, drug, num_queries=num_queries))

    structured_outputs_params = StructuredOutputsParams(json=QUERY_TRIALS_SCHEMA)
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, seed=SEED, structured_outputs=structured_outputs_params)
    raw_outputs = batch_chat(model, sampling_params, prompts)
    
    all_trials = []
    num_duplicate_trials = []
    grouped_raw_outputs = []
    grouped_prompts = []
    output_idx = 0
    separator = f"\n\n{'=' * 200}\n\n"
    start_time = time.perf_counter()

    for count in counts:
        patient_raw_outputs = raw_outputs[output_idx : output_idx + count]
        patient_prompts = prompts[output_idx : output_idx + count]
        patient_all_trials = []
        patient_duplicates = []
        
        for response, prompt in zip(patient_raw_outputs, patient_prompts):
            try:
                result = orjson.loads(response)
            except orjson.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse JSON. The model likely hit a repetition loop and was truncated.\n"
                    f"JSON Error: {e}\n\n"
                    f"Prompt:\n{prompt}\n\n"
                    f"Raw Model Output:\n{response}"
                ) from e

            queries = result.get("queries")

            if queries is None:
                raise ValueError(f"""
                    Queries is missing\n
                    Here is the prompt:\n\n
                    {prompt}\n\n
                    Here is the output:\n\n
                    {response}
                    """
                )

            trials = search_clinical_trials(
                queries, 
                hasResults=True, 
                studyType="INTERVENTIONAL", 
                output_path=output_path
            )

            unique_num_trials, total_num_trials = count_num_trials(trials)
            patient_duplicates.append((unique_num_trials, total_num_trials))
            patient_all_trials.append(trials)

        grouped_raw_outputs.append(separator.join(patient_raw_outputs))
        grouped_prompts.append(separator.join([orjson.dumps(p, option=orjson.OPT_INDENT_2).decode("utf-8") for p in patient_prompts]))

        all_trials.append(patient_all_trials)
        num_duplicate_trials.append(patient_duplicates)
        output_idx += count

    elapsed = time.perf_counter() - start_time
    
    return all_trials, grouped_raw_outputs, grouped_prompts, num_duplicate_trials, elapsed

EVALUATE_RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "2-3 sentences explaining for why this relevance score was assigned. Do not use markdown, XML, HTML, or tags."
        },
        "relevanceScore": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "An integer score from 0 to 100 indicating how relevant this trial is to the patient."
        }
    },
    "required": [
        "reasoning",
        "relevanceScore"
    ],
    "additionalProperties": False
}

def generate_trials_task(model: ActorPool, patient_notes: list[dict[str, str]], trials: list[list[dict[str, Any]]], output_path: Path, rewrite: bool = False) -> tuple[dict[str, dict[str, Any]], list[str], list[str], float]:
    
    def __process__(patient_note: dict[str, str], trial: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        note_id = patient_note.get("source", {}).get("note_id")
        nct_id = trial.get("nctId")
        
        if not nct_id:
            raise ValueError(f"\nNo nctID is found. Here is the trial: {trial}\n")
            
        trial_content_path = str(output_path / f"{nct_id}.json")
        meta_info = trial.get("meta")

        if meta_info is None:
            raise ValueError(f"Trial {nct_id} has no metadata.")
        
        rxclass = meta_info.get("rxclass")
        drug = meta_info.get("drug")
        rela = meta_info.get("rela")
        query_dict = meta_info.get("query")
        
        if not rxclass or not drug:
            raise ValueError(f"""
                No RxClass or Drug found.\n
                - Note ID: {note_id}\n
                - NCT ID: {nct_id}\n
                - Attached meta object: {meta_info}
                """
            )
        
        provenance_entry = {
            "extraction": str(note_id),
            "rxclass": rxclass,
            "drug": drug,
            "rela": rela,
            "query": query_dict
        }
        
        meta_dict = {
            "nct_id": nct_id,
            "note_id": str(note_id),
            "trial_path": trial_content_path,
            "provenance": provenance_entry
        }
        
        prompt = generate_trials_prompt(patient_note, trial)
        return nct_id, meta_dict, prompt

    def __evaluate__(target_data: dict[str, Any], meta: dict[str, Any], response: str, prompt: list[dict[str, str]]) -> None:
        note_id = meta["note_id"]
        try:
            result = orjson.loads(response)
        except orjson.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON. The model likely hit a repetition loop and was truncated.\n"
                f"JSON Error: {e}\n\n"
                f"Prompt:\n{prompt}\n\n"
                f"Raw Model Output:\n{response}"
            ) from e
        
        relevance_score = result.get("relevanceScore")
        reasoning = result.get("reasoning")

        if relevance_score is None or reasoning is None:
            raise ValueError(f"""
                relevance_score or reasoning is missing.\n
                Relevance_score: {relevance_score}\n
                Reasoning: {reasoning}\n
                Here is the prompt:\n\n
                {prompt}\n\n
                Here is the output:\n\n
                {response}
                """
            )
        
        if note_id in target_data["notes"]:
            idx = target_data["notes"].index(note_id)
            if rewrite:
                target_data["provenance"][idx] = meta["provenance"]
                target_data["relevance"][idx] = relevance_score
                target_data["reasoning"][idx] = reasoning
        else:
            target_data["provenance"].append(meta["provenance"])
            target_data["notes"].append(note_id)
            target_data["relevance"].append(relevance_score)
            target_data["reasoning"].append(reasoning)

    prompts = []
    metadata = []
    counts = []
    
    for i, patient_note in enumerate(patient_notes):
        patient_trials = trials[i]
        counts.append(len(patient_trials))
        
        for trial in patient_trials:
            _, meta_dict, prompt = __process__(patient_note, trial)
            metadata.append(meta_dict)
            prompts.append(prompt)

    structured_outputs_params = StructuredOutputsParams(json=EVALUATE_RELEVANCE_SCHEMA)
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, seed=SEED, structured_outputs=structured_outputs_params)
    start_time = time.perf_counter()
    raw_outputs = batch_chat(model, sampling_params, prompts)
    elapsed = time.perf_counter() - start_time
    
    grouped_raw_outputs = []
    grouped_prompts = []
    output_idx = 0
    separator = f"\n\n{'=' * 200}\n\n"

    for count in counts:
        patient_raw_outputs = raw_outputs[output_idx : output_idx + count]
        patient_prompts = prompts[output_idx : output_idx + count]

        grouped_raw_outputs.append(separator.join(patient_raw_outputs))
        grouped_prompts.append(separator.join([orjson.dumps(p, option=orjson.OPT_INDENT_2).decode("utf-8") for p in patient_prompts]))

        output_idx += count
    
    aggregated_results = {}
    
    for meta, response, prompt in zip(metadata, raw_outputs, prompts):
        nct_id = meta["nct_id"]
        file_path = output_path / f"{nct_id}.json"
        
        if nct_id not in aggregated_results:
            if file_path.exists():
                with file_path.open("rb") as f:
                    orjson.loads(f.read())
                    if "reasoning" not in data:
                        data["reasoning"] = []
                    aggregated_results[nct_id] = data
            else:
                aggregated_results[nct_id] = {
                    "trial": meta["trial_path"],
                    "provenance": [],
                    "notes": [],
                    "relevance": [],
                    "reasoning": []
                }
                
        __evaluate__(aggregated_results[nct_id], meta, response, prompt)
        
    for nct_id, data in aggregated_results.items():
        file_path = output_path / f"{nct_id}.json"
        with file_path.open("w", encoding="utf-8") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            
    return aggregated_results, grouped_raw_outputs, grouped_prompts, elapsed

if __name__ == "__main__":
    print(len(RXCLASS_NAMES))