from tools.RxClass import getClassTypeDescription, getRelaDescription
import json
from pathlib import Path

RXCLASS_DRUGS_PATH = Path("/sc/arion/projects/EHR_ML/lia38/Evidence_Gap_Clinical_Trials/data/RxClass")

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
                {chr(10).join(f"- {name}" for name in subclasses)}
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

.               I'm going to give you a list of RxClass Class Types and list of 
                classes within each class type. You must choose only one class that
                is relevant to the patient notes. The reason we are doing this is because
                after you choose a class, we will find all the drug members within that 
                class, find trials relates to the drug and its RxClass information, and 
                evaluate how relevant the trials are to the patient notes. You must call the tool `select_rxclass` 
                exactly once by inputting exactly how the class is written. 
                For example, if there is a class called 'Abdomen, Acute' then output exactly as written, 'Abdomen, Acute'. 
                
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
            "content": ("""
                You are an expert clinical trial retrieval assistant responsible
                for building a drug-to-clinical-trial knowledge graph.
            """
            ),
        },
        {
            "role": "user",
            "content": f"""
                Read the information below about this patient
                
                {json.dumps(patient_note)}

                Drug Information of {drug_name} with RxClass Information

                RxClass:
                {className}

                Relationship:
                {rela}

                A description on how the drug relates to the class:
                {getRelaDescription(relaSource, rela)}

                Generate {num_queries} queries on ClinicalTrial.gov to obtain
                as many trials as possible that relates to this drug and its RxClass information. 
                You will call search_clinical_trials exactly once with a parameter called queries 
                that is a list of dict, each with the keys 'term', 'condition', and 'intervention'. 
                One search on ClinicalTrial.gov uses a term, condition, and intervention. Depending on the 
                relationship of the drug, you may not need to use a field. 
                Try a wide range of queries to minimize retrieving duplicated trials. Use different names
                of the same drug, like brand names. Do not generate duplicate queries, meaning no two queries 
                should be the exact same. Provide your reasoning (or uncertainity) for choosing these specific queries. 
                """
        },
    ]

    return messages

# def suggest_trial_queries(rxcui: str, className: str, rela: str, relaSource: str, num_queries: int = 5) -> list[dict[str, str]]:
#     drug_name = getRxConceptProperties(rxcui).get("name")
#     messages = [
#         {
#             "role": "system",
#             "content": ("""
#                 You are an expert clinical trial retrieval assistant responsible
#                 for building a drug-to-clinical-trial knowledge graph.
#             """
#             ),
#         },
#         {
#             "role": "user",
#             "content": f"""
#                 Drug Information of {drug_name} with RxClass Information

#                 RxClass:
#                 {className}

#                 Relationship:
#                 {rela}

#                 A description on how the drug relates to the class:
#                 {getRelaDescription(relaSource, rela)}

#                 Generate {num_queries} queries on ClinicalTrial.gov to obtain
#                 as many trials as possible that relates to this drug and its RxClass information. 
#                 You will call search_clinical_trials exactly once with a parameter called queries 
#                 that is a list of dict, each with the keys 'term', 'condition', and 'intervention'. 
#                 One search on ClinicalTrial.gov uses a term, condition, and intervention. Depending on the 
#                 relationship of the drug, you may not need to use a field. 
#                 Try a wide range of queries to minimize retrieving duplicated trials. Use different names
#                 of the same drug, like brand names. Do not generate duplicate queries, meaning no two queries 
#                 should be the exact same. Provide your reasoning (or uncertainity) for choosing these specific queries. 
#                 """
#         },
#     ]

#     return messages

# EVALUATE_TRIAL_TOOL = {
#     "type": "object",
#     "properties": {
#         "relevanceScore": {
#             "type": "integer",
#             "minimum": 0,
#             "maximum": 100,
#             "description": (
#                 "Overall relevance score of the clinical trial to the drug (0-100). "
#                 "0 means completely unrelated; 100 means directly studies the drug."
#             ),
#         },
#         "reasoning": {
#             "type": "string",
#             "description": "Detailed explanation for why this relevance score was assigned.",
#         },
#         "queries": {
#             "type": "array",
#             "description": "List of new, diverse search query objects for ClinicalTrials.gov.",
#             "items": {
#                 "type": "object",
#                 "properties": {
#                     "term": {
#                         "type": ["string", "null"],
#                         "description": "Free-text keyword search or null.",
#                     },
#                     "condition": {
#                         "type": ["string", "null"],
#                         "description": "Condition or disease name or null.",
#                     },
#                     "intervention": {
#                         "type": ["string", "null"],
#                         "description": "Intervention or treatment name or null.",
#                     },
#                 },
#                 "required": ["term", "condition", "intervention"],
#                 "additionalProperties": False,
#             },
#         },
#     },
#     "required": ["relevanceScore", "reasoning", "queries"],
#     "additionalProperties": False,
# }

# def evaluate_trial(trial_path: Path, rxcui: str, className: str, rela: str, relaSource: str, previous_queries: list[dict[str, Any]], num_queries: int=16) -> list[dict[str, str]]:
#     trial = json.loads(trial_path.read_text())
#     drug_name = getRxConceptProperties(rxcui).get("name")
#     messages = [
#         {
#             "role": "system",
#             "content": ("""
#                 You are an expert clinical trial retrieval assistant responsible
#                 for building a drug-to-clinical-trial knowledge graph.
#             """
#             ),
#         },
#         {
#             "role": "user",
#             "content": f"""
#                 Drug Information of {drug_name} with RxClass Information

#                 RxClass:
#                 {className}

#                 Relationship:
#                 {rela}

#                 A description on how the drug relates to the class:
#                 {getRelaDescription(relaSource, rela)}

#                 Queries used on ClinicalTrial.gov to pull these trials

#                 {json.dumps(previous_queries)}

#                 Here is content of one clinical trial from the previous queries

#                 {json.dumps(trial)}

#                 1. Read and understand what were the goals and results of this trial. You will also call the tool call
#                evaluate_trial_tool once.

#                 2. Assign a "relevanceScore", an integer between 0 and 100 that represents
#                 how relevant this trial is to the drug and its RxClass Information. For example,
#                     - 0 = completely unrelated to the drug and its RxClass info.
#                     - 50 = partially related or only indirectly studies the drug and its RxClass info.
#                     - 100 = directly studies the drug and its RxClass info and should definitely be linked.
                
#                 3. Explain your reasoning for assigning this score in the "reasoning" field. 
#                 Do not consider factors such as the phase of the trial, when the trial finished, or 
#                 if the trial is completed. How you decide on the score should be based on more important 
#                 factors on the trial, such as the cohorts, inclusion and exclusion criteria, what were the metrics, and what 
#                 were the results. Whether or not the results are successful or not does not matter. 

#                 4. Generate {num_queries} new, diverse search query objects under `queries` to find additional relevant trials. 
#                     - Ensure queries are different from 'Queries Already Used'.
#                     - Each query object must contain 'term', 'condition', and 'intervention' (use null if not applicable).
#                     - These queries should covers issues that could have increase the relevanceScore. 
#                     Or, if the relevanceScore is high, find more relevant trials. 
#                     - Each new query generated should not be exactly the same to each other
#             """
#         }
#     ]

#     return message