from pathlib import Path
from vllm import SamplingParams
from ray.util import ActorPool
from tools.Ray import get_actor_pool
from tools.tasks import select_rxclass_task
from tools.RxClass import getClassesMembers
import json
from datetime import datetime
from typing import Any
import time

PATIENT_NOTES_PATH = Path("/sc/arion/projects/EHR_ML/sgelman/evidence_gap/data/extractions")
TRIAL_STORE_PATH = Path("/sc/arion/projects/EHR_ML/lia38/Evidence_Gap_Clinical_Trials/data/clinical_trials")
RESPONSES_STORE_PATH = Path("/sc/arion/projects/EHR_ML/lia38/Evidence_Gap_Clinical_Trials/data/responses")

def log(patient_notes: list[dict[str, str]], message: str, contents: list[Any] | None = None, stage: str | None = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prefix = f"[{timestamp}]"
    if stage:
        prefix += f" [{stage}]"

    if contents is not None and len(contents) != len(patient_notes):
        raise ValueError(
            "contents must have the same length as patient_notes."
        )

    for i, patient_note in enumerate(patient_notes):
        patient_id = str(patient_note.get("source").get("note_id"))
        log_path = RESPONSES_STORE_PATH / patient_id / "pipeline.log"

        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{prefix} {message}\n")

            if contents is not None:
                f.write(f"{prefix} {contents[i]}\n")

def dump(patient_notes: list[dict[str, str]], raw_outputs: list[str]) -> None:
    if len(raw_outputs) != len(patient_notes):
        raise ValueError(
            "outputs must have the same length as patient_notes."
        )

    for patient_note, output in zip(patient_notes, raw_outputs):
        patient_id = str(patient_note.get("source").get("note_id"))
        raw_path = RESPONSES_STORE_PATH / patient_id / "raw_outputs.txt"

        with raw_path.open("a", encoding="utf-8") as f:
            f.write(output)
            f.write("\n")

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
    log(patient_notes, "Starting inital Nemotron task", stage="RXCLASS")
    selected_classes, raw_outputs, elapsed = select_rxclass_task(model, sampling_params, patient_notes)
    log(patient_notes, "Successfully asked Nemotron to read the extraction and choose a RxClass", contents=selected_classes, stage="RXCLASS")
    log(patient_notes, f"This task took an average time of {(elapsed / len(patient_notes)):.2f} seconds.", stage="RXCLASS")
    dump(patient_notes, raw_outputs)
    log(patient_notes, f"Updated raw_output.txt", stage="RXCLASS")

    log(patient_notes, f"Fetching drug members from RxClass API", stage="DRUGS")
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
    log(patient_notes, "Successfully retrieved drug members", contents=drug_members, stage="DRUGS")
    log(patient_notes, f"This task took an average time of {(elapsed / len(patient_notes)):.2f} seconds.", stage="DRUGS")



if __name__ == "__main__":
    patient_notes = load_patient_notes(PATIENT_NOTES_PATH, limit=1)
    log(patient_notes, "Successfully loaded patient notes.", stage="LOAD")
    pool = get_actor_pool()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100000)
    process_notes(patient_notes, pool, sampling_params)