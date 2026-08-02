from pathlib import Path
from ray.util import ActorPool
from tools.Ray import get_actor_pool
from tools.tasks import select_rxclass_task, query_trials_task, generate_trials_task, SELECT_RXCLASS_SCHEMA
from tools.RxClass import getClassesMembers
from tools.ClinicalTrialGov import get_nct_id, extract_clinical_info
from tools.logger import log, dump, RESPONSES_STORE_PATH
import orjson
import time
from typing import Any

PATIENT_NOTES_PATH = Path("/sc/arion/projects/EHR_ML/sgelman/evidence_gap/data/extractions")
TRIAL_STORE_PATH = Path("/sc/arion/projects/EHR_ML/lia38/Evidence_Gap_Clinical_Trials/data/clinical_trials")

def load_patient_notes(notes_dir: Path, limit: int | None = None) -> list[dict[str, str]]:
    patient_notes = []

    for file_path in sorted(notes_dir.glob("*.json"))[:limit]:
        with file_path.open("rb") as f:
            patient_note = orjson.loads(f.read())

        patient_id = str(patient_note["source"]["note_id"])

        output_dir = RESPONSES_STORE_PATH / "patients" / patient_id
        output_dir.mkdir(parents=True, exist_ok=True)

        log_path = output_dir / "pipeline.log"
        raw_path = output_dir / "raw_outputs.txt"
        prompts_path = output_dir / "prompts.txt"

        log_path.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)
        prompts_path.unlink(missing_ok=True)

        log_path.touch()
        raw_path.touch()
        prompts_path.touch()

        patient_notes.append(patient_note)

    return patient_notes

def select_rxclasses(model: ActorPool, patient_notes: list[dict[str, str]]) -> list[dict[str, Any]]:
    log(patient_notes, "Starting inital Nemotron task.", stage="RXCLASS")
    selected_classes, raw_outputs, prompt_outputs, elapsed = select_rxclass_task(model, patient_notes)
    log(patient_notes, "Successfully asked Nemotron to read the extraction and choose a RxClass.", contents=selected_classes, stage="RXCLASS")
    log(patient_notes, f"This task took an average time of {(elapsed / len(patient_notes)):.2f} seconds.", stage="RXCLASS")
    dump(patient_notes, prompt_outputs, filename="prompts.txt")
    log(patient_notes, "Updated prompts.txt.", stage="RXCLASS")
    dump(patient_notes, raw_outputs)
    log(patient_notes, "Updated raw_output.txt.", stage="RXCLASS")
    return selected_classes

def fetch_drug_members(patient_notes: list[dict[str, str]], selected_classes: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    log(patient_notes, f"Fetching drug members from RxClass API.", stage="DRUGS")
    class_ids = [cla["classId"] for cla in selected_classes]
    class_names = [cla["className"] for cla in selected_classes]
    class_types = [cla["classType"] for cla in selected_classes]
    
    start_time = time.perf_counter()
    drug_members = getClassesMembers(
        classIds=class_ids,
        classNames=class_names,
        classTypes=class_types,
        extend=True,
        ttys=["IN"],
        trans=1,
        ignoreNoneRela=True
    )
    elapsed = time.perf_counter() - start_time
    
    for patient_note, members in zip(patient_notes, drug_members):
        log(
            [patient_note],
            f"Successfully retrieved {len(members)} drug members.",
            contents=[members],
            stage="DRUGS",
        )
    log(patient_notes, f"This task took an average time of {(elapsed / len(patient_notes)):.2f} seconds.", stage="DRUGS")
    return drug_members


def query_trials(model: ActorPool, patient_notes: list[dict[str, str]], drug_members: list[list[dict[str, Any]]]):
    log(patient_notes, "Starting ClinicalTrials.gov query generation task.", stage="TRIALS")
    
    trials, query_raw_outputs, query_prompts, num_duplicate_trials, query_elapsed = query_trials_task(
        model, 
        patient_notes, 
        drug_members, 
        TRIAL_STORE_PATH,
        num_queries=3
    )

    trial_counts_summary = []
    for duplicates in num_duplicate_trials:
        raw_total = sum(total for _, total in duplicates)
        unique_total = sum(unique for unique, _ in duplicates)
        summary_msg = f"Retrieved in total {raw_total} trials ({unique_total} total unique trials across search queries) from ClinicalTrials.gov."
        trial_counts_summary.append(summary_msg)

    log(patient_notes, "Successfully queried clinical trials.", contents=trial_counts_summary, stage="TRIALS")
    log(patient_notes, "For each drug from RxClass, recorded the aggregated unique and total trial counts for each evaluated drug (unique_num_trials, total_num_trials):", contents=num_duplicate_trials, stage="TRIALS")
    log(patient_notes, f"The time it took to query ClinicalTrials.gov took an average time of {(query_elapsed / len(patient_notes)):.2f} seconds.", stage="TRIALS")
    dump(patient_notes, query_prompts, filename="prompts.txt")
    log(patient_notes, "Updated prompts.txt.", stage="TRIALS")
    dump(patient_notes, query_raw_outputs)
    log(patient_notes, "Updated raw_output.txt.", stage="TRIALS")
    
    return trials


def evaluate_trials(model: ActorPool, patient_notes: list[dict[str, str]], all_trials, drug_members):
    log(patient_notes, "Starting clinical trial relevance evaluation task.", stage="EVALUATE")

    formatted_trials = []

    for i, patient_drug_queries in enumerate(all_trials):
        patient_studies = []
        seen_ncts = set()
        patient_drugs = drug_members[i]
        
        for drug_query_results, drug_member in zip(patient_drug_queries, patient_drugs):
            drug_name = drug_member.get("minConcept", {}).get("name")
            rxclass_name = drug_member.get("classInfo", {}).get("className")
            rela = drug_member.get("classInfo", {}).get("rela")
            
            if not drug_name or not rxclass_name:
                raise ValueError(f"""
                        Missing drug_name and rxclass_name\n
                        Drug Member Object: {drug_member}   
                    """
                )

            for query_res in drug_query_results:
                query_dict = query_res.get("query")
                for raw_study in query_res.get("studies", []):
                    nct_id = get_nct_id(raw_study)
                    
                    if nct_id and nct_id not in seen_ncts:
                        seen_ncts.add(nct_id)
                        extracted_study = extract_clinical_info(raw_study)
                        extracted_study["meta"] = {
                            "rxclass": rxclass_name,
                            "drug": drug_name,
                            "rela": rela,  
                            "query": query_dict
                        }
                        patient_studies.append(extracted_study)
                        
        formatted_trials.append(patient_studies)

    eval_results, eval_raw_outputs, eval_prompts, eval_elapsed = generate_trials_task(
        model=model,
        patient_notes=patient_notes,
        trials=formatted_trials,
        output_path=RESPONSES_STORE_PATH,
        rewrite=True
    )
    
    log(patient_notes, "Successfully evaluated clinical trials relevance.", stage="EVALUATE")
    log(patient_notes, f"This task took an average time of {(eval_elapsed / len(patient_notes)):.2f} seconds.", stage="EVALUATE")
    dump(patient_notes, eval_prompts, filename="prompts.txt")
    log(patient_notes, "Updated prompts.txt.", stage="EVALUATE")
    dump(patient_notes, eval_raw_outputs)
    log(patient_notes, "Updated raw_output.txt.", stage="EVALUATE")
    
    return eval_results

def process_patient_notes(patient_notes: list[dict[str, str]], model: ActorPool) -> None:
    selected_classes = select_rxclasses(model, patient_notes)
    drug_members = fetch_drug_members(patient_notes, selected_classes)
    all_trials = query_trials(model, patient_notes, drug_members)
    eval_results = evaluate_trials(model, patient_notes, all_trials, drug_members)

if __name__ == "__main__":
    patient_notes = load_patient_notes(PATIENT_NOTES_PATH)
    log(patient_notes, "Successfully loaded patient notes.", stage="LOAD")
    pool = get_actor_pool()
    process_patient_notes(patient_notes, pool)