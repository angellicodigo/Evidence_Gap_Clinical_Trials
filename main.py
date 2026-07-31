from pathlib import Path
from vllm import SamplingParams
from ray.util import ActorPool
from tools.Ray import get_actor_pool
from tools.tasks import select_rxclass_task, query_trials_task, generate_trials_task
from tools.RxClass import getClassesMembers
from tools.ClinicalTrialGov import get_nct_id, extract_clinical_info
from tools.logger import log, dump, RESPONSES_STORE_PATH
import json
import time

PATIENT_NOTES_PATH = Path("/sc/arion/projects/EHR_ML/sgelman/evidence_gap/data/extractions")
TRIAL_STORE_PATH = Path("/sc/arion/projects/EHR_ML/lia38/Evidence_Gap_Clinical_Trials/data/clinical_trials")

def load_patient_notes(notes_dir: Path, limit: int | None = None) -> list[dict[str, str]]:
    patient_notes = []

    for file_path in sorted(notes_dir.glob("*.json"))[:limit]:
        with file_path.open("r", encoding="utf-8") as f:
            patient_note = json.load(f)

        patient_id = str(patient_note["source"]["note_id"])

        output_dir = RESPONSES_STORE_PATH / patient_id
        output_dir.mkdir(parents=True, exist_ok=True)

        log_path = output_dir / "pipeline.log"
        raw_path = output_dir / "raw_outputs.txt"

        log_path.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)

        log_path.touch()
        raw_path.touch()

        patient_notes.append(patient_note)

    return patient_notes


def process_notes(patient_notes: list[dict[str, str]], model: ActorPool, sampling_params: SamplingParams) -> None:
    log(patient_notes, "Starting inital Nemotron task.", stage="RXCLASS")
    selected_classes, raw_outputs, elapsed = select_rxclass_task(model, sampling_params, patient_notes)
    log(patient_notes, "Successfully asked Nemotron to read the extraction and choose a RxClass.", contents=selected_classes, stage="RXCLASS")
    log(patient_notes, f"This task took an average time of {(elapsed / len(patient_notes)):.2f} seconds.", stage="RXCLASS")
    dump(patient_notes, raw_outputs, )
    log(patient_notes, f"Updated raw_output.txt.", stage="RXCLASS")

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
    log(patient_notes, f"Successfully retrieved {len(drug_members)} drug members.", contents=drug_members, stage="DRUGS")
    log(patient_notes, f"This task took an average time of {(elapsed / len(patient_notes)):.2f} seconds.", stage="DRUGS")

    log(patient_notes, "Starting ClinicalTrials.gov query generation task.", stage="TRIALS")
    all_trials, num_duplicate_trials, query_raw_outputs, query_elapsed = query_trials_task(
        model, 
        sampling_params, 
        patient_notes, 
        drug_members, 
        TRIAL_STORE_PATH
    )
    trial_counts_summary = []
    for i, patient_note in enumerate(patient_notes):
        p_id = patient_note.get("source", {}).get("note_id")
        # num_duplicate_trials[i] contains tuples of (unique_num_trials, total_num_trials) for each drug
        raw_total = sum(total for _, total in num_duplicate_trials[i])
        unique_total = sum(unique for unique, _ in num_duplicate_trials[i])
        
        summary_msg = f"Retrieved {raw_total} raw trials ({unique_total} unique across search queries) from ClinicalTrials.gov."
        print(f"[TRIALS STAGE] Patient Note {p_id}: {summary_msg}")
        trial_counts_summary.append(summary_msg)

    log(patient_notes, "Successfully queried clinical trials.", contents=trial_counts_summary, stage="TRIALS")
    log(patient_notes, "For each query, we record a tuple (unique_num_trials, total_num_trials)", contents=num_duplicate_trials, stage="TRIALS")
    log(patient_notes, f"The time it took to query ClinicalTrials.gov took an average time of {(query_elapsed / len(patient_notes)):.2f} seconds.", stage="TRIALS")
    dump(patient_notes, query_raw_outputs)
    log(patient_notes, "Updated raw_output.txt.", stage="TRIALS")

    log(patient_notes, "Starting clinical trial relevance evaluation task.", stage="EVALUATE")
    
    # Format and deduplicate trials per patient note
    formatted_trials = []
    for patient_drug_queries in all_trials:
        patient_studies = []
        seen_ncts = set()
        
        for drug_query_results in patient_drug_queries:
            for query_res in drug_query_results:
                for raw_study in query_res.get("studies", []):
                    # Get the NCT ID directly using the helper function
                    nct_id = get_nct_id(raw_study)
                    
                    if nct_id and nct_id not in seen_ncts:
                        seen_ncts.add(nct_id)
                        # Extract full clinical information for Nemotron evaluation
                        patient_studies.append(extract_clinical_info(raw_study))
                        
        formatted_trials.append(patient_studies)

    eval_results, eval_raw_outputs, eval_elapsed = generate_trials_task(
        model=model,
        sampling_params=sampling_params,
        patient_notes=patient_notes,
        trials=formatted_trials,
        output_path=RESPONSES_STORE_PATH
    )
    
    log(patient_notes, "Successfully evaluated clinical trials relevance.", stage="EVALUATE")
    log(patient_notes, f"This evaluation task took an average time of {(eval_elapsed / len(patient_notes)):.2f} seconds.", stage="EVALUATION")
    dump(patient_notes, eval_raw_outputs)
    log(patient_notes, "Updated raw_output.txt.", stage="EVALUATE")

if __name__ == "__main__":
    patient_notes = load_patient_notes(PATIENT_NOTES_PATH, limit=1)
    log(patient_notes, "Successfully loaded patient notes.", stage="LOAD")
    pool = get_actor_pool()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100000)
    process_notes(patient_notes, pool, sampling_params)