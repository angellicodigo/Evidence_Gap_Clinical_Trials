import pandas as pd
from typing import Any
from pathlib import Path
from Evidence_Gap_Clinical_Trials.tools.ClinicalTrialGov import search_clinical_trials, SEARCH_CLINICAL_TRIALS_TOOL
from Evidence_Gap_Clinical_Trials.tools.prompts import suggest_trial_queries, evaluate_trial, EVALUATE_TRIAL_TOOL
from ray.util import ActorPool
from vllm import SamplingParams
from Evidence_Gap_Clinical_Trials.tools.parse_tool_call import parse_tool_call

def __chunk_into_batches__(content: list, batch_size: int) -> list[Any]:
    return [content[i:i + batch_size] for i in range(0, len(content), batch_size)]

def initialize_drugs(drugs_path: Path) -> tuple[dict, dict, dict, list]:
    # Reads parquets and sets up drug states and file storage
    drug_states = {}
    class_data_results = {}
    class_reasoning_logs = {}
    initial_prompts_queue = []
    
    for file_path in drugs_path.glob("*.parquet"):
        class_name = file_path.stem
        df = pd.read_parquet(file_path)
        
        class_data_results[class_name] = []
        class_reasoning_logs[class_name] = []

        for _, row in df.iterrows():
            rxcui = row['RXCUI']
            prompt = suggest_trial_queries(rxcui, class_name, row['Class Relation'], row['Relation Source'])
            
            drug_states[(class_name, rxcui)] = {
                "rela": row['Class Relation'],
                "rela_source": row['Relation Source'],
                "previous_queries": [],
                "seen_nctids": set(),
                "trials_to_evaluate": []
            }
            initial_prompts_queue.append((prompt, class_name, rxcui))
            
    return drug_states, class_data_results, class_reasoning_logs, initial_prompts_queue

def run_initial_queries(model: ActorPool, sampling_params: SamplingParams, drug_states: dict, initial_prompts_queue: list, batch_size: int, trial_store_path: Path) -> None:
    # Executes initial vLLM batch generation and fetches clinical trials
    prompts_only = [item[0] for item in initial_prompts_queue]
    batches = __chunk_into_batches__(prompts_only, batch_size)
    
    worker_outputs = list(model.map(
        lambda actor, batch: actor.chat.remote(messages=batch, sampling_params=sampling_params, tools=[SEARCH_CLINICAL_TRIALS_TOOL]), 
        batches
    ))

    flat_vllm_outputs = [out for w_out in worker_outputs for out in w_out.get("result", [])]
    
    for idx, request_output in enumerate(flat_vllm_outputs):
        _, class_name, rxcui = initial_prompts_queue[idx]
        drug_state = drug_states[(class_name, rxcui)]
        response_text = request_output.outputs[0].text
        tool_params = parse_tool_call(response_text)
        queries = tool_params.get('queries')
        drug_state["previous_queries"].extend(queries)
        
        search_outputs = search_clinical_trials(
            queries=queries, 
            hasResults=True, 
            studyType='INTERVENTIONAL', 
            output_path=trial_store_path, 
            excludeDuplicates=True
        )
        
        for result in search_outputs:
            for path in result.get('paths', []):
                if path.stem not in drug_state["seen_nctids"]:
                    drug_state["seen_nctids"].add(path.stem)
                    drug_state["trials_to_evaluate"].append(path)

def evaluate_trials(model: ActorPool, sampling_params: SamplingParams, drug_states: dict, class_data_results: dict, class_reasoning_logs: dict, batch_size: int, max_iterations: int, trial_store_path: Path) -> None:
    # Loops through iterations to evaluate trials and fetch new ones based on suggestions
    for _ in range(1, max_iterations + 1):
        eval_prompts_queue = []
        
        for (class_name, rxcui), drug_state in drug_states.items():
            trials_this_round = drug_state["trials_to_evaluate"]
            drug_state["trials_to_evaluate"] = [] 
            
            for trial_path in trials_this_round:
                prompt = evaluate_trial(
                    trial_path=trial_path, rxcui=rxcui, className=class_name, 
                    rela=drug_state["rela"], relaSource=drug_state["rela_source"], previous_queries=drug_state["previous_queries"]
                )
                eval_prompts_queue.append((prompt, class_name, rxcui, trial_path))
        
        if not eval_prompts_queue:
            break
                        
        eval_prompts_only = [item[0] for item in eval_prompts_queue]
        eval_batches = __chunk_into_batches__(eval_prompts_only, batch_size)
        
        eval_worker_outputs = list(model.map(
            lambda actor, batch: actor.chat.remote(messages=batch, sampling_params=sampling_params, tools=[EVALUATE_TRIAL_TOOL]), 
            eval_batches
        ))
        
        flat_eval_outputs = [out for w_out in eval_worker_outputs for out in w_out.get("result", [])]
        
        for idx, request_output in enumerate(flat_eval_outputs):
            _, class_name, rxcui, trial_path = eval_prompts_queue[idx]
            drug_state = drug_states[(class_name, rxcui)]
            nctid = trial_path.stem
            
            eval_response_text = request_output.outputs[0].text
            eval_params = parse_tool_call(eval_response_text)
            
            score = eval_params.get('relevanceScore', 0)
            reasoning = eval_params.get('reasoning', 'No reasoning provided.')
            
            class_data_results[class_name].append({
                "rxcui": rxcui,
                "nctid": nctid,
                "relevanceScore": score,
                "reasoning": reasoning
            })
            class_reasoning_logs[class_name].append(
                f"===== RXCUI: {rxcui} | NCTID: {nctid} =====\n{eval_response_text}\n\n"
            )
            
            new_queries = eval_params.get('queries')
            drug_state["previous_queries"].extend(new_queries)
            
            new_search_outputs = search_clinical_trials(
                queries=new_queries, 
                hasResults=True, 
                studyType='INTERVENTIONAL', 
                output_path=trial_store_path, 
                excludeDuplicates=True, 
                excludeNctIds=list(drug_state["seen_nctids"]) 
            )
            
            for new_res in new_search_outputs:
                for new_path in new_res.get('paths', []):
                    if new_path.stem not in drug_state["seen_nctids"]:
                        drug_state["seen_nctids"].add(new_path.stem)
                        drug_state["trials_to_evaluate"].append(new_path)

def save_results(output_dir: Path, class_data_results: dict, class_reasoning_logs: dict) -> None:
    """Saves the dataframes and reasoning text files to disk."""
    for class_name, rows in class_data_results.items():
        if rows:
            res_df = pd.DataFrame(rows)
            res_df.to_parquet(output_dir / f"{class_name}.parquet", index=False)
            
        logs = class_reasoning_logs.get(class_name, [])
        if logs:
            with open(output_dir / f"{class_name}.txt", "w", encoding="utf-8") as f:
                f.writelines(logs)