from tools.RxClass import getClassTypeDescription, getRelaDescription
import json
from pathlib import Path
from typing import Any

RXCLASS_DRUGS_PATH = Path("/sc/arion/projects/EHR_ML/lia38/Evidence_Gap_Clinical_Trials/data/RxClass")

def select_rxclass_prompt(patient_note: dict[str, str]) -> list[dict[str, str]]:
    sections = []

    for txt_file in sorted(RXCLASS_DRUGS_PATH.glob("*.txt")):
        class_type = txt_file.stem

        description = getClassTypeDescription(class_type)

        subclasses = []
        with txt_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Each line: <Class Name>\t<ClassId>
                class_name = line.split("\t", 1)[0]
                subclasses.append(class_name)

        sections.append(
            f"""Class Type: {class_type}

                Description of the Class Type:
                {description}

                Available Classes:
                {"\n".join(f"- {name}" for name in subclasses)}
        """
        )

    sections_str = "\n\n".join(sections)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert clinical trial retrieval assistant responsible "
                "for building a drug-to-clinical-trial knowledge graph."
            ),
        },
        {
            "role": "user",
            "content": f"""
                Read the information below about this patient
                
                {json.dumps(patient_note)}

.               Here is a list of RxClass Class Types and list of 
                classes within each class type. You must choose only one class that
                is relevant to the patient notes. The reason we are doing this is because
                after you choose a class, we will find all the drug members within that 
                class, find trials relates to the drug and its RxClass information, and 
                evaluate how relevant the trials are to the patient notes. You must call the tool `select_rxclass` 
                exactly once by inputting exactly how the class is written. 
                For example, if there is a class called 'Abdomen, Acute' then output exactly as written, 'Abdomen, Acute'. 
                Do not invent or infer class names outside the provided list. Double check that your choice of the class name 
                appears verbatim in one of the Available Classes (for example, Lyme disease is not in the list). 
                
                {sections_str}
        """
        },
    ]

    return messages

def query_trials_prompt(patient_note: dict[str, str], drug_member: dict[str, str], num_queries: int = 5) -> list[dict[str, str]]:
    drug_name = drug_member.get("minConcept").get("name")
    className = drug_member.get("classInfo").get("className")
    rela = drug_member.get("classInfo").get("rela")
    relaSource = drug_member.get("classInfo").get("relaSource")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert clinical trial retrieval assistant responsible "
                "for building a drug-to-clinical-trial knowledge graph."
            ),
        },
        {
            "role": "user",
            "content": f"""
                Read the information below about this patient:
                
                {json.dumps(patient_note)}

                Drug Information for {drug_name} with RxClass Information:

                RxClass:
                {className}

                Relationship:
                {rela}

                Description of how the drug relates to the class:
                {getRelaDescription(relaSource, rela)}

                Generate {num_queries} queries for ClinicalTrial.gov to obtain
                as many trials as possible related to this drug and its RxClass information. 

                You must call `search_clinical_trials` exactly once with a parameter named `queries`. 
                `queries` must be a list of objects where each object contains the keys: 'term', 'condition', and 'intervention'. 

                Instructions:
                1. If a field is not needed for a specific query, set its value strictly to `null` (e.g., "intervention": null).
                2. Do NOT use Python's `None`, single quotes, or omit any keys.
                3. Try a wide range of queries to minimize duplicate trials (use brand names, alternate terms, etc.).
                4. Do not generate duplicate queries, every query object must be unique.
                """,
        },
    ]

    return messages

def generate_trials_prompt(patient_note: dict[str, str], trial: dict[str, Any]) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert clinical trial retrieval assistant responsible "
                "for evaluating the relevance between a patient's clinical note and a clinical trial."
            ),
        },
        {
            "role": "user",
            "content": f"""
                Evaluate the relevance of the following clinical trial to the patient's clinical note.

                Patient Note:
                {json.dumps(patient_note)}

                Clinical Trial Content:
                {json.dumps(trial)}
                1. Read and understand what were the goals and results of this trial. You will also call the tool call
               evaluate_trial_tool once.

                2. Assign a "relevanceScore", an integer between 0 and 100 that represents
                how relevant this trial is to the drug and its RxClass Information. For example,
                    - 0 = completely unrelated to the drug and its RxClass info.
                    - 50 = partially related or only indirectly studies the drug and its RxClass info.
                    - 100 = directly studies the drug and its RxClass info and should definitely be linked.
                
                3. Explain your reasoning for assigning this score in the "reasoning" field. 
                Do not consider factors such as the phase of the trial, when the trial finished, or 
                if the trial is completed. How you decide on the score should be based on more important 
                factors on the trial, such as the cohorts, inclusion and exclusion criteria, what were the metrics, and what 
                were the results. Whether or not the results are successful or not does not matter. 

                1. Read and understand the patient's condition, the trial's goals, inclusion/exclusion criteria, and results of the trials.
                2. Assign a "relevanceScore", an integer between 0 and 100 that represents how relevant this trial is to the patient.
                    - 0 = completely unrelated
                    - 50 = partially related
                    - 100 = highly relevant
                3. Explain your reasoning for assigning this score in the "reasoning" field.
                4. Call the tool `evaluate_relevance` exactly once.
            """
        }
    ]
    return messages

if __name__ == "__main__":
    dummy_patient_note = {
        "source": {"note_id": "166098869"},
        "text": "The patient mentions cannabis abuse, cyclic vomiting syndrome, gastroparesis, opioid dependence, dysautonomia, rheumatoid arthritis, diarrhea, cough, and chest tightness."
    }

    
    messages = select_rxclass_prompt(dummy_patient_note)
    print(messages)
        
