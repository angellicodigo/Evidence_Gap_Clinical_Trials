from tools.RxClass import getRelaDescription
import json
from pathlib import Path
from typing import Any

RXCLASS_DRUGS_PATH = Path("/sc/arion/projects/EHR_ML/lia38/Evidence_Gap_Clinical_Trials/data/RxClass")

def select_rxclass_prompt(patient_note: dict[str, str]) -> list[dict[str, str]]:
    sections = []
    index = 0

    for txt_file in sorted(RXCLASS_DRUGS_PATH.glob("*.txt")):
        class_type = txt_file.stem

        numbered_classes = []

        with txt_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                class_name = line.split("\t", 1)[0]
                numbered_classes.append(f"{index}: {class_name}")
                index += 1

        sections.append(
            f"""Class Type: {class_type}

                Available Classes:
                {"\n".join(numbered_classes)}
        """
        )

    sections_str = "\n\n".join(sections)

    example = {
        "reasoning": "The patient presents with writing difficulty and left upper extremity coordination deficits, with occupational therapy focusing on improving hand dexterity and shoulder function. The primary need appears related to motor retraining and rehabilitation of fine motor skills. Among the RxClass disease list, Cerebral Palsy (class index 102) is a condition often requiring similar therapeutic interventions for motor coordination and upper extremity rehabilitation, making it the closest match for the patient's clinical needs.",
        "index": 102
    }

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
                First, read and understand the patient information below.

                {json.dumps(patient_note)}

                # Core Responsibilities
                - Evaluate the patient's notes
                - Identify and review up to 3 candidate RxClass entries from the list below that best match the patient's primary clinical needs.
                - Rank these candidates from best to worst based on clinical relevance.
                - Select the single highest-ranked candidate index for final output 

                # Guidelines
                ## Clinical Reasoning & Selection
                - Do not consider more than 3 candidates. These candidates can be in different classtypes or in the same one.
                - If an exact match is missing, use your best clinical judgment to pick the closest alternative.
                - Rely on the RxClass list strictly as reference material. Do not copy, output, echo, or scan the list in your reasoning. 

                ## Output Constraints
                - Do not repeat, quote, summarize, or enumerate the RxClass list in your response.
                - Return a JSON object that conforms **exactly** to the provided schema. 

                - `reasoning`
                    - Explicitly name the chosen RxClass and its index
                    - Provide a concise clinical justification in 2-3 sentences explaining why the selected RxClass best matches the patient's primary clinical needs.
                    - Do **not** mention your internal ranking, candidate list, or reasoning process.
                    - Do **not** quote or enumerate the RxClass reference list.
                    - Do **not** include XML, HTML, Markdown, code fences, or special tags.

                - `index`
                    - Return the zero-based integer index of the selected RxClass.
                    - The index must correspond to an entry in the provided RxClass reference list.

                - Do not return any fields other than `reasoning` and `index`.

                # Example
                {json.dumps(example)}

                Full RxClass Information:

                {sections_str}
            """
        }
    ]

    return messages

def query_trials_prompt(patient_note: dict[str, str], drug_member: dict[str, str], num_queries: int = 5) -> list[dict[str, str]]:
    drug_name = drug_member.get("minConcept").get("name")
    className = drug_member.get("classInfo").get("className")
    rela = drug_member.get("classInfo").get("rela")
    relaSource = drug_member.get("classInfo").get("relaSource")

    example = {
        "reasoning": (
            "The search strategy focuses primarily on the patient's disease and "
            "the selected drug while broadening coverage using synonymous disease "
            "terms and related interventions. This maximizes recall while reducing "
            "duplicate clinical trial results."
        ),
        "queries": [
            {
                "term": None,
                "condition": "Crohn Disease",
                "intervention": "ustekinumab",
            },
            {
                "term": None,
                "condition": "Inflammatory Bowel Disease",
                "intervention": "Stelara",
            }
        ],
    }

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
                Read the patient information below.

                {json.dumps(patient_note)}

                Drug Information

                Drug:
                {drug_name}

                RxClass:
                {className}

                Relationship:
                {rela}

                Description:
                {getRelaDescription(relaSource, rela)}

                # Core Responsibilities
                - Evaluate the patient information.
                - Use the drug and RxClass information to construct {num_queries} ClinicalTrials.gov search queries.
                - Maximize recall while minimizing duplicate studies.

                # Guidelines
                ## Search Strategy
                - Generate exactly {num_queries} unique query objects.
                - Use disease synonyms, drug synonyms, brand names, biomarkers, or related terminology when appropriate.
                - Place each search term in the most appropriate field.

                ## Output Constraints
                - Return a JSON object that conforms exactly to the provided schema.
                - If a field is unnecessary, set it to null.
                - Every query object must contain the keys:
                    - term
                    - condition
                    - intervention
                - Do not generate duplicate query objects.
                - Do not include XML, HTML, Markdown, code fences, or special tags.

                # Example
                {json.dumps(example, indent=2)}
            """
        },
    ]

    return messages

def generate_trials_prompt(patient_note: dict[str, str], trial: dict[str, Any]) -> list[dict[str, str]]:
    example = {
        "reasoning": (
            "The trial closely matches the patient's primary diagnosis and "
            "eligibility profile. Although some secondary characteristics do not "
            "perfectly align, the intervention and study objectives remain highly relevant."
        ),
        "relevanceScore": 91,
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert clinical trial retrieval assistant responsible "
                "for evaluating the relevance between a patient's clinical note "
                "and a clinical trial."
            ),
        },
        {
            "role": "user",
            "content": f"""
                First, read and understand the patient information below.

                Patient Note

                {json.dumps(patient_note)}

                Clinical Trial

                {json.dumps(trial)}

                # Core Responsibilities
                - Evaluate the patient's clinical characteristics.
                - Evaluate the clinical trial.
                - Determine how relevant this trial is for the patient.
                - Produce a single relevance score between 0 and 100.

                # Guidelines
                ## Clinical Reasoning
                - Consider the patient's diagnoses, treatments, demographics, and important clinical history.
                - Carefully evaluate the trial objectives, intervention, eligibility criteria, cohorts, endpoints, and reported results.
                - Do not consider the trial phase, completion status, or whether the study was successful.
                - Focus on clinical relevance rather than study quality.

                ## Output Constraints
                - Return a JSON object that conforms exactly to the provided schema.

                - reasoning
                    - Provide a concise clinical justification in 2-3 sentences.
                    - Explain why the assigned relevance score is appropriate.
                    - Do not mention your internal reasoning process.
                    - Do not include XML, HTML, Markdown, code fences, or special tags.

                - relevanceScore
                    - Return an integer between 0 and 100.
                    - 0 = completely unrelated
                    - 50 = partially relevant
                    - 100 = highly relevant

                - Do not return any fields other than reasoning and relevanceScore.

                # Example
                {json.dumps(example, indent=2)}
        """
        }
    ]

    return messages