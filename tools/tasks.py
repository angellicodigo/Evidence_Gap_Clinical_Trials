import time
from typing import Any
from ray.util import ActorPool
from vllm import SamplingParams
from tools.parse_tool_call import parse_tool_call
from tools.prompts import select_rxclass_prompt, RXCLASS_DRUGS_PATH, query_trials_prompt, generate_trials_prompt
from tools.ClinicalTrialGov import search_clinical_trials, count_num_trials
from pathlib import Path
import json

RXCLASS_MAPPING = {}
for txt_file in RXCLASS_DRUGS_PATH.glob("*.txt"):
    class_type = txt_file.stem
    with txt_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, class_id = line.split("\t", 1)
            
            # Check for cross-file duplicates to replicate your previous RuntimeError check
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

def batch_chat(model: ActorPool, sampling_params: SamplingParams, prompts: list[list[dict[str, str]]], tools: list[dict[str, Any]] | None = None, batch_size: int = 64) -> list[str]:
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    
    worker_outputs = list(model.map(
        lambda actor, batch: actor.chat.remote(messages=batch, sampling_params=sampling_params, tools=tools),
        batches
    ))

    # Flatten nested outputs into a single list of response texts
    responses = []

    for worker_out in worker_outputs:
        for req_out in worker_out.get("result", []):
            responses.append(req_out.outputs[0].text)

    return responses

SELECT_RXCLASS_TOOL = {
    "type": "function",
    "function": {
        "name": "select_rxclass",
        "description": "Select the most relevant RxClass.",
        "parameters": {
            "type": "object",
            "properties": {
                "class": {
                    "type": "string",
                    "description": "The exact name of the selected RxClass class."
                }
            },
            "required": ["class"],
            "additionalProperties": False,
        },
    },
}

def select_rxclass_task(model: ActorPool, sampling_params: SamplingParams, patient_notes: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str], float]:
    prompts = []
    for patient_note in patient_notes:
        prompts.append(select_rxclass_prompt(patient_note))

    start_time = time.perf_counter()
    raw_outputs = batch_chat(
        model,
        sampling_params,
        prompts,
        [SELECT_RXCLASS_TOOL]
    )
    elapsed = time.perf_counter() - start_time

    selected_classes = []

    for i, response in enumerate(raw_outputs):
        tool_params = parse_tool_call(response)
        class_name = tool_params.get("class")

        # Use the O(1) dictionary lookup instead of nested loops
        match = RXCLASS_MAPPING.get(class_name)

        if not match:
            # Capture the associated patient ID to know exactly who caused it
            patient_id = patient_notes[i].get("source", {}).get("note_id", "Unknown")
            
            raise FileNotFoundError(f"""
                RxClass '{class_name}' does not exist in any RxClass .txt file\n.
                Patient ID '{patient_id}' caused an error.\n
                Nemotron Raw Output/Reasoning:\n{response}
                """ 
            )

        selected_classes.append(match)

    return selected_classes, raw_outputs, elapsed

SEARCH_CLINICAL_TRIALS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_clinical_trials",
        "description": ("""
            Search for clinical studies from ClinicalTrials.gov using one or more
            search queries. Each query object may specifc a term, condition, and/or
            intervention. Only use English spellings and double check your spelling.
            Put each search term into the filter where it fits the best. For example, 
            put a disease term in the Condition/disease filter and a drug term in the 
            Intervention/treatment filter. Limit the number of search terms you use in each filter.
            Start by filling in the filters for the information most important to you
            """
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "description": "A list of search query objects.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "term": {
                                "type": ["string", "null"],
                                "description": ("""
                                        This filter is a tool to help focus your search for clinical studies 
                                        using terms that don't fit in the other basic filters. For example, 
                                        you can use this filter to search for studies with a specific word in 
                                        the study title or study description. Other examples of terms you can 
                                        use in this field include:

                                        - The NCT number
                                        - The acronym for a study
                                        - A biomarker or gene name
                                        - An outcome measure
                                                
                                        Avoid using this filter to type in several different search terms or 
                                        terms that fit better in the other basic filters.
                                                
                                        Note that the term field searches everything. If you search for "cancer" for term, you'll 
                                        get trials where cancer appears in the title, conditions, interventions, description, 
                                        eligibility criteria -- everywhere.
                                                
                                        Use null if no free-text term is needed.
                                    """
                                )
                            },
                            "condition": {
                                "type": ["string", "null"],
                                "description": ("""
                                    Use this field to search for studies related to a condition, disease, disorder, 
                                    syndrome, illness, or injury (For example, breast cancer or high blood pressure). 
                                    If you are searching for studies about more than one condition, 
                                    enter each condition separately.
                                                
                                    Use null if no condition is specified.
                                    """
                                )
                            },
                            "intervention": {
                                "type": ["string", "null"],
                                "description": ("""
                                    Use this field to search for studies that use a specific drug, 
                                    medical device, procedure, or lifestyle change (For example, 
                                    radiation therapy or low-fat diet).
                                                
                                    Use null if no intervention is specified.
                                    """
                                )
                            }
                        },
                        "additionalProperties": False
                    }
                }
            },
            "required": ["queries"],
        },
    },
}

def query_trials_task(model: ActorPool, sampling_params: SamplingParams, patient_notes: list[dict[str, str]], drug_members: list[list[dict[str, Any]]], output_path: Path) -> tuple[list[list[list[dict[str, Any]]]], list[list[tuple[int, int]]], list[str], float]:
    prompts = []
    # Keeps track of how many drugs per RxClass 
    drug_counts = []
    
    # Map every drug to a prompt, tracking counts per patient
    for i in range(len(patient_notes)):
        patient_drugs = drug_members[i]
        drug_counts.append(len(patient_drugs))
        for drug in patient_drugs:
            prompts.append(query_trials_prompt(patient_notes[i], drug))

    raw_outputs = batch_chat(
        model,
        sampling_params,
        prompts,
        [SEARCH_CLINICAL_TRIALS_TOOL]
    )
    
    all_trials = []
    num_duplicate_trials = []
    grouped_raw_outputs = []
    output_idx = 0

    # Re-group outputs and query results by patient
    start_time = time.perf_counter()

    for count in drug_counts:
        # List slicing to grab the chunk of responses belonging to the patient
        patient_raw_outputs = raw_outputs[output_idx : output_idx + count]
        patient_all_trials = []
        patient_duplicates = []
        
        for response in patient_raw_outputs:
            tool_params = parse_tool_call(response)
            queries = tool_params.get("queries")

            if not queries:
                raise ValueError(
                    f"The LLM did not use the tool call correctly. Here is the output:\n{response}"
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

        # This part of the code is formatting the output for dump()
        separator = f"\n\n{'=' * 200}\n\n"
        grouped_raw_outputs.append(separator.join(patient_raw_outputs))
        all_trials.append(patient_all_trials)
        num_duplicate_trials.append(patient_duplicates)
        
        output_idx += count

    elapsed = time.perf_counter() - start_time
    
    return all_trials, num_duplicate_trials, grouped_raw_outputs, elapsed

EVALUATE_RELEVANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "evaluate_relevance",
        "description": "Evaluate the relevance of a clinical trial to a patient note.",
        "parameters": {
            "type": "object",
            "properties": {
                "relevanceScore": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "An integer score from 0 to 100 indicating how relevant this trial is to the patient."
                },
                "reasoning": {
                    "type": "string",
                    "description": "Detailed explanation for why this relevance score was assigned."
                }
            },
            "required": ["relevanceScore", "reasoning"],
            "additionalProperties": False,
        },
    },
}

def generate_trials_task(model: ActorPool, sampling_params: SamplingParams, patient_notes: list[dict[str, str]], trials: list[list[dict[str, Any]]], output_path: Path, rewrite: bool = False) -> tuple[dict[str, dict[str, Any]], list[str], float]:
    
    def __process__(patient_note: dict[str, str], trial: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        note_id = patient_note.get("source", {}).get("note_id")
        nct_id = trial.get("nctId")
        
        if not nct_id:
            raise ValueError(f"\nNo nctID is found. Here is the trial: {trial}\n")
            
        trial_content_path = str(output_path / f"{nct_id}.json")
        meta_info = trial.get("meta")
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
        
        providence_entry = {
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
            "providence": providence_entry
        }
        
        prompt = generate_trials_prompt(patient_note, trial)
        return nct_id, meta_dict, prompt

    def __evaluate__(target_data: dict[str, Any], meta: dict[str, Any], response: str) -> None:
        note_id = meta["note_id"]
        tool_params = parse_tool_call(response)
        relevance_score = tool_params.get("relevanceScore")
        reasoning = tool_params.get("reasoning")
        
        # Checks if the current patient note has already been evaluated
        # in this trial json before
        if note_id in target_data["notes"]:
            idx = target_data["notes"].index(note_id)
            if rewrite:
                target_data["providence"][idx] = meta["providence"]
                target_data["relevance"][idx] = relevance_score
                target_data["reasoning"][idx] = reasoning
        else:
            target_data["providence"].append(meta["providence"])
            target_data["notes"].append(note_id)
            target_data["relevance"].append(relevance_score)
            target_data["reasoning"].append(reasoning)

    prompts = []
    # Stores a tracking dictionary for each trial-patient pair (NCT ID, note ID, path, and providence)
    metadata = []
    trial_counts = []
    
    # Prepare prompts and track metadata using helper
    for i, patient_note in enumerate(patient_notes):
        patient_trials = trials[i]
        trial_counts.append(len(patient_trials))
        
        for trial in patient_trials:
            # The helper function validates the trial, extracts its metadata, and generates its evaluation prompt
            _, meta_dict, prompt = __process__(patient_note, trial)
            metadata.append(meta_dict)
            prompts.append(prompt)
    
    start_time = time.perf_counter()

    raw_outputs = batch_chat(
        model, 
        sampling_params, 
        prompts, 
        [EVALUATE_RELEVANCE_TOOL]
    )

    elapsed = time.perf_counter() - start_time
    
    # Group raw outputs per patient note
    # Prepares a list to store concatenated raw text outputs mapped 1-to-1 with each patient note
    grouped_raw_outputs = []
    output_idx = 0
    for count in trial_counts:
        patient_raw_outputs = raw_outputs[output_idx : output_idx + count]
        separator = f"\n\n{'=' * 200}\n\n"
        grouped_raw_outputs.append(separator.join(patient_raw_outputs))
        output_idx += count
    
    # Aggregate results and load/save trial data files
    aggregated_results = {}
    
    for meta, response in zip(metadata, raw_outputs):
        nct_id = meta["nct_id"]
        file_path = output_path / f"{nct_id}.json"
        
        if nct_id not in aggregated_results:
            if file_path.exists():
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "reasoning" not in data:
                        data["reasoning"] = []
                    aggregated_results[nct_id] = data
            else:
                aggregated_results[nct_id] = {
                    "trial": meta["trial_path"],
                    "providence": [],
                    "notes": [],
                    "relevance": [],
                    "reasoning": []
                }
                
        __evaluate__(aggregated_results[nct_id], meta, response)
        
    # Write all updated records back to disk
    for nct_id, data in aggregated_results.items():
        file_path = output_path / f"{nct_id}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
    return aggregated_results, grouped_raw_outputs, elapsed