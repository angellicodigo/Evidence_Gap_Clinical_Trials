from datetime import datetime
from typing import Any
from pathlib import Path

RESPONSES_STORE_PATH = Path("/sc/arion/projects/EHR_ML/lia38/Evidence_Gap_Clinical_Trials/data/responses")

def log(patient_notes: list[dict[str, str]], message: str, contents: list[Any] | None = None, stage: str | None = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prefix = f"[{timestamp}]"
    if stage:
        prefix += f" [{stage}]"

    if contents is not None and len(contents) != len(patient_notes):
        raise ValueError(f"""
            Length mismatch: len(contents) is {len(contents)}, but len(patient_notes) is {len(patient_notes)}.
            """
        )

    for i, patient_note in enumerate(patient_notes):
        patient_id = str(patient_note.get("source").get("note_id"))
        log_path = RESPONSES_STORE_PATH / patient_id / "pipeline.log"

        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{prefix} {message}\n")

            if contents is not None:
                f.write(f"{prefix} {contents[i]}\n")

def dump(patient_notes: list[dict[str, str]], outputs: list[str], filename: str = "raw_outputs.txt") -> None:
    if len(outputs) != len(patient_notes):
        raise ValueError(
            "outputs must have the same length as patient_notes."
        )

    for patient_note, output in zip(patient_notes, outputs):
        patient_id = str(patient_note.get("source").get("note_id"))
        file_path = RESPONSES_STORE_PATH / patient_id / filename

        with file_path.open("a", encoding="utf-8") as f:
            f.write(output)
            f.write("\n")