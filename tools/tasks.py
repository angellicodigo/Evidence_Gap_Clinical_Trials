import time
from typing import Any
from ray.util import ActorPool
from vllm import SamplingParams
from tools.parse_tool_call import parse_tool_call
from tools.prompts import select_rxclass_prompt, SELECT_RXCLASS_TOOL, RXCLASS_DRUGS_PATH

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

def batch_chat(model: ActorPool, sampling_params: SamplingParams, prompts: list[list[dict[str, str]]], tools: list[dict[str, Any]] | None = None, batch_size: int = 256) -> list[str]:
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


def select_rxclass_task(model: ActorPool, sampling_params: SamplingParams, patient_notes: list[dict[str, str]]):
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

    for response in raw_outputs:
        tool_params = parse_tool_call(response)
        class_name = tool_params["class"]

        # Use the O(1) dictionary lookup instead of nested loops
        match = RXCLASS_MAPPING.get(class_name)

        if not match:
            raise FileNotFoundError(
                f"RxClass '{class_name}' does not exist in any RxClass .txt file."
            )

        selected_classes.append(match)

    return selected_classes, raw_outputs, elapsed

SEARCH_CLINICAL_TRIALS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_clinical_trials",
        "description": (
            "Search completed clinical studies from the ClinicalTrials.gov API "
            "using one or more search queries. Each query object may specify a "
            "free-text term, a condition, and/or an intervention."
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
                                "description": (
                                    "Free-text search terms. Examples: "
                                    "'lung cancer', 'EGFR', 'stage IV melanoma'. "
                                    "Use null if no free-text term is needed."
                                )
                            },
                            "condition": {
                                "type": ["string", "null"],
                                "description": (
                                    "Disease or condition names. Examples: "
                                    "'Non-Small Cell Lung Cancer', "
                                    "'Type 2 Diabetes', 'Breast Cancer'. "
                                    "Use null if no condition is specified."
                                )
                            },
                            "intervention": {
                                "type": ["string", "null"],
                                "description": (
                                    "Intervention or treatment names. Examples: "
                                    "'Pembrolizumab', 'Osimertinib', "
                                    "'CAR-T', 'Radiation Therapy'. "
                                    "Use null if no intervention is specified."
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

def query_trials_task(model: ActorPool, sampling_params: SamplingParams, patient_notes: list[dict[str, str]], drug_members: list[list[dict[str, Any]]]):
    prompts = []
    for drug_member in drug_members:
        prompts.append(query_trials_prompt(drug_member))

    start_time = time.perf_counter()
    raw_outputs = batch_chat(
        model,
        sampling_params,
        prompts,
        [SELECT_RXCLASS_TOOL]
    )
    elapsed = time.perf_counter() - start_time

    selected_classes = []

    for response in raw_outputs:
        tool_params = parse_tool_call(response)
        class_name = tool_params["class"]

        # Use the O(1) dictionary lookup instead of nested loops
        match = RXCLASS_MAPPING.get(class_name)

        if not match:
            raise FileNotFoundError(
                f"RxClass '{class_name}' does not exist in any RxClass .txt file."
            )

        selected_classes.append(match)

    return selected_classes, raw_outputs, elapsed

# def query_trials_task(class_name: str, rela_source: str = "MEDRT") -> list[dict[str, Any]]:

#     members_dict = getClassMembers(classId=class_id, relaSource=rela_source)
    
#     return [
#         {
#             "rxcui": concept.get("rxcui"),
#             "name": concept.get("name"),
#             "rela": rela,
#             "relaSource": rela_source
#         }
#         for rela, member_list in members_dict.items()
#         for item in member_list
#         if (concept := item.get("minConcept")) and concept.get("rxcui")
#     ]


# def generate_queries_task(
#     model: ActorPool,
#     sampling_params: SamplingParams,
#     drugs: list[dict[str, Any]],
#     class_name: str,
#     patient_note: str,
#     batch_size: int = 64
# ) -> tuple[dict[str, dict[str, Any]], list[str], float]:
#     """
#     Task 3: Generates 5 search queries per drug in parallel batches across Ray actors.
#     Returns: (drug_queries_map, list_of_raw_llm_responses, elapsed_time)
#     """
#     start_time = time.perf_counter()
#     prompts = [
#         prompt_suggest_drug_queries(
#             drug_name=d["name"], rxcui=d["rxcui"], class_name=class_name, patient_note=patient_note, num_queries=5
#         )
#         for d in drugs
#     ]

#     raw_responses = run_batch_chat(
#         model=model,
#         sampling_params=sampling_params,
#         prompts=prompts,
#         tools=[SEARCH_CLINICAL_TRIALS_TOOL],
#         batch_size=batch_size
#     )

#     drug_queries_map = {}
#     for drug, raw_text in zip(drugs, raw_responses):
#         parsed = parse_tool_call(raw_text)
#         queries = parsed.get("queries", [])[:5] # Restrict to exactly 5 queries
#         drug_queries_map[drug["rxcui"]] = {
#             "drug_name": drug["name"],
#             "queries": queries
#         }

#     elapsed = time.perf_counter() - start_time
#     return drug_queries_map, raw_responses, elapsed


# def search_trials_task(
#     drug_queries_map: dict[str, dict[str, Any]],
#     trial_store_path: Path
# ) -> tuple[dict[str, Path], list[str], float]:
#     """
#     Task 4: Pulls clinical trials sequentially (max_workers=1) and builds metrics.
#     Returns: (unique_trials_map, metric_logs, elapsed_time)
#     """
#     start_time = time.perf_counter()
#     unique_trials: dict[str, Path] = {}
#     metric_logs: list[str] = []

#     total_pulled_overall = 0
#     total_duplicates_overall = 0

#     for rxcui, info in drug_queries_map.items():
#         queries = info["queries"]
#         drug_name = info["drug_name"]
#         if not queries:
#             continue

#         # Sequential trial fetching (max_workers=1 disables parallel thread pool execution)
#         search_results = search_clinical_trials(
#             queries=queries,
#             hasResults=True,
#             studyType="INTERVENTIONAL",
#             output_path=trial_store_path,
#             excludeDuplicates=False,
#             max_workers=1 
#         )

#         drug_pulled = 0
#         drug_seen_ncts = set()

#         for q_idx, res in enumerate(search_results):
#             q_studies = res.get("studies", [])
#             q_paths = res.get("paths", [])
#             drug_pulled += len(q_studies)

#             q_duplicates = 0
#             for path in q_paths:
#                 nctid = path.stem
#                 if nctid in drug_seen_ncts:
#                     q_duplicates += 1
#                 else:
#                     drug_seen_ncts.add(nctid)
#                     unique_trials[nctid] = path

#             metric_logs.append(
#                 f"[METRICS] Drug: {drug_name} | Query #{q_idx+1}: {queries[q_idx]} -> "
#                 f"Pulled: {len(q_studies)} | Duplicates in query: {q_duplicates}"
#             )

#         drug_unique_count = len(drug_seen_ncts)
#         drug_duplicates = drug_pulled - drug_unique_count
#         total_pulled_overall += drug_pulled
#         total_duplicates_overall += drug_duplicates

#         metric_logs.append(
#             f"--> [DRUG SUMMARY] {drug_name}: Pulled = {drug_pulled}, "
#             f"Unique = {drug_unique_count}, Duplicates = {drug_duplicates}\n"
#         )

#     elapsed = time.perf_counter() - start_time
#     metric_logs.append(
#         f"[SEARCH OVERALL] Pulled = {total_pulled_overall} | "
#         f"Unique = {len(unique_trials)} | Duplicates = {total_duplicates_overall}"
#     )

#     return unique_trials, metric_logs, elapsed


# def evaluate_trials_task(
#     model: ActorPool,
#     sampling_params: SamplingParams,
#     unique_trials: dict[str, Path],
#     patient_note: str,
#     patient_note_id: str,
#     drug_name: str,
#     batch_size: int = 64
# ) -> tuple[list[dict[str, Any]], list[str], float]:
#     """
#     Task 5: Evaluates retrieved clinical trials in parallel batches via Ray.
#     Returns: (list_of_eval_results, list_of_raw_llm_responses, elapsed_time)
#     """
#     start_time = time.perf_counter()
#     nctids = list(unique_trials.keys())
    
#     prompts = [
#         prompt_evaluate_trial(
#             json.loads(unique_trials[nctid].read_text(encoding="utf-8")),
#             drug_name,
#             patient_note
#         )
#         for nctid in nctids
#     ]

#     if not prompts:
#         return [], [], 0.0

#     raw_responses = run_batch_chat(
#         model=model,
#         sampling_params=sampling_params,
#         prompts=prompts,
#         tools=[EVALUATE_TRIAL_TOOL],
#         batch_size=batch_size
#     )

#     evaluations = []
#     for nctid, raw_text in zip(nctids, raw_responses):
#         trial_data = json.loads(unique_trials[nctid].read_text(encoding="utf-8"))
#         parsed = parse_tool_call(raw_text)

#         evaluations.append({
#             "nctid": nctid,
#             "json_data": {
#                 "trial": trial_data.get("briefTitle") or trial_data.get("officialTitle") or "Unknown Title",
#                 "provenance": "",
#                 "patient_note_id": patient_note_id,
#                 "relevanceScore": parsed.get("relevanceScore", 0),
#                 "reasoning": parsed.get("reasoning", "No reasoning provided.")
#             }
#         })

#     elapsed = time.perf_counter() - start_time
#     return evaluations, raw_responses, elapsed


# # =====================================================================
# # Side-Effect Layer (IO, Logging, and Persistence)
# # =====================================================================

# def persist_results_and_logs(
#     evaluations: list[dict[str, Any]],
#     raw_logs: list[str],
#     metric_logs: list[str],
#     output_dir: Path,
#     log_path: Path
# ) -> None:
#     """
#     Isolated Side-Effect Function: Handles disk writes for log files, raw output texts, and output JSON files.
#     """
#     output_dir.mkdir(parents=True, exist_ok=True)
#     log_path.parent.mkdir(parents=True, exist_ok=True)

#     # Write trial output JSON files: filename is the nctid
#     for item in evaluations:
#         nctid = item["nctid"]
#         out_path = output_dir / f"{nctid}.json"
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(item["json_data"], f, indent=2)

#     # Append metrics and raw logs
#     with open(log_path, "a", encoding="utf-8") as f:
#         f.write("\n=== METRICS LOGS ===\n")
#         for line in metric_logs:
#             f.write(line + "\n")

#         f.write("\n=== RAW NEMOTRON LOGS ===\n")
#         for raw in raw_logs:
#             f.write(raw + "\n" + "=" * 60 + "\n")
        
#         f.flush()