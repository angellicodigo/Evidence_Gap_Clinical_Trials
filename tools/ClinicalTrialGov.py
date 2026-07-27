from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

retry_strategy = Retry(
    total=8,                          # up to 8 retries per request
    backoff_factor=1,                 # sleeps 1s, 2s, 4s, 8s, 16s...
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True,  # honor the server's Retry-After header if it sends one
)

max_connections = 64
session = requests.Session()
adapter = HTTPAdapter(pool_connections=max_connections, pool_maxsize=max_connections)
session.mount("https://", adapter)
session.mount("http://", adapter)

def extract_clinical_info(study: dict[str, Any]) -> dict[str, Any]:
    protocolSection = study.get("protocolSection", {})
    identificationModule = protocolSection.get("identificationModule", {})
    statusModule = protocolSection.get("statusModule", {})
    descriptionModule = protocolSection.get("descriptionModule", {})
    conditionsModule = protocolSection.get("conditionsModule", {})
    designModule = protocolSection.get("designModule", {})
    eligibilityModule = protocolSection.get("eligibilityModule", {})
    armsInterventionsModule = protocolSection.get("armsInterventionsModule", {})
    designInfo = designModule.get("designInfo", {})
    resultsSection = study.get("resultsSection", {})

    extracted = {
        "nctId": identificationModule.get("nctId"),
        "briefTitle": identificationModule.get("briefTitle"),
        "officialTitle": identificationModule.get("officialTitle"),
        "status": {
            "overallStatus": statusModule.get("overallStatus"),
            "startDate": statusModule.get("startDateStruct", {}).get("date"),
            "lastUpdatedPostDate": statusModule.get("lastUpdatePostDateStruct", {}).get("date"),
        },
        "description": {
            "briefSummary": descriptionModule.get("briefSummary"),
            "detailedDescription": descriptionModule.get("detailedDescription"),
        },
        "conditions": conditionsModule.get("conditions", []),
        "design": {
            "studyType": designModule.get("studyType"),
            "phases": designModule.get("phases"),
            "allocation": designInfo.get("allocation"),
            "interventionModel": designInfo.get("interventionModel"),
            "interventionModelDescription": designInfo.get("interventionModelDescription"),
            "primaryPurpose": designInfo.get("primaryPurpose"),
            "observationalModel": designInfo.get("observationalModel"),
            "timePerspective": designInfo.get("timePerspective"),
            "maskingInfo": designInfo.get("maskingInfo", {}),
            "enrollmentInfo": designModule.get("enrollmentInfo", {}),
        },
        "eligibility": {
            "eligibilityCriteria": eligibilityModule.get("eligibilityCriteria"),
            "healthyVolunteers": eligibilityModule.get("healthyVolunteers"),
            "sex": eligibilityModule.get("sex"),
            "genderBased": eligibilityModule.get("genderBased"),
            "genderDescription": eligibilityModule.get("genderDescription"),
            "minimumAge": eligibilityModule.get("minimumAge"),
            "maximumAge": eligibilityModule.get("maximumAge"),
            "stdAges": eligibilityModule.get("stdAges"),
            "studyPopulation": eligibilityModule.get("studyPopulation"),
            "samplingMethod": eligibilityModule.get("samplingMethod"),
        },
        "interventions": [
            {
                "type": intervention.get("type"),
                "name": intervention.get("name"),
                "description": intervention.get("description"),
                "otherNames": intervention.get("otherNames", []),
            }
            for intervention in armsInterventionsModule.get("interventions", [])
        ],
        "armGroups": [
            {
                "label": arm.get("label"),
                "type": arm.get("type"),
                "description": arm.get("description"),
                "interventionNames": arm.get("interventionNames", []),
            }
            for arm in armsInterventionsModule.get("armGroups", [])
        ],
    }

    if resultsSection:
        participantFlowModule = resultsSection.get("participantFlowModule", {})
        baselineCharacteristicsModule = resultsSection.get("baselineCharacteristicsModule", {})
        outcomeMeasuresModule = resultsSection.get("outcomeMeasuresModule", {})
        adverseEventsModule = resultsSection.get("adverseEventsModule", {})
        moreInfoModule = resultsSection.get("moreInfoModule", {})

        extracted["results"] = {
            "participantFlow": {
                "preAssignmentDetails": participantFlowModule.get("preAssignmentDetails"),
                "recruitmentDetails": participantFlowModule.get("recruitmentDetails"),
                "groups": participantFlowModule.get("groups", []),
            },
            "baselineCharacteristics": {
                "populationDescription": baselineCharacteristicsModule.get("populationDescription"),
                "groups": baselineCharacteristicsModule.get("groups", []),
                "denoms": baselineCharacteristicsModule.get("denoms", []),
                "measures": baselineCharacteristicsModule.get("measures", []),
            },
            "outcomesMeasures": [
                {
                    k: v
                    for k, v in outcome.items()
                    if k != "reportingStatus"
                }
                for outcome in outcomeMeasuresModule.get("outcomeMeasures", [])
            ],
            "adverseEvents": {
                "frequencyThreshold": adverseEventsModule.get("frequencyThreshold"),
                "timeFrame": adverseEventsModule.get("timeFrame"),
                "description": adverseEventsModule.get("description"),
                "allCauseMortalityComment": adverseEventsModule.get("allCauseMortalityComment"),
                "eventGroups": adverseEventsModule.get("eventGroups", []),
                "seriousEvents": adverseEventsModule.get("seriousEvents", []),
                "otherEvents": adverseEventsModule.get("otherEvents", []),
            },
            "limitationsAndCaveats": moreInfoModule.get("limitationsAndCaveats", {}),
        }

    return extracted


def download_clinical_trials(
    studies: list[dict[str, Any]],
    output_path: Path,
    redownload: bool = False
) -> list[Path]:
    paths = []

    for study in studies:
        nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")

        if not nct_id:
            continue

        output_file = output_path / f"{nct_id}.json"

        if output_file.exists() and not redownload:
            paths.append(output_file)
            continue

        extracted = extract_clinical_info(study)

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2)

        paths.append(output_file)

    return paths

def search_clinical_trials(
    queries: list[dict[str, str | None]],
    patient_info: str | None = None,
    hasResults: bool = False,
    studyType: str | None = None,
    excludeNctIds: list[str] | None = None,
    output_path: Path | None = None,
    excludeDuplicates: bool = False,
    num_workers: int = 64
) -> list[dict[str, Any]]:

    if not queries:
        raise ValueError("The 'queries' parameter cannot be empty.")

    def __search_query__(
        query: dict[str, str | None],
        patient_info: str | None = None,
        hasResults: bool = False,
        studyType: str | None = None,
        excludeNctIds: list[str] | None = None,
        output_path: Path | None = None
    ) -> dict[str, Any]:
        
        fields = ",".join([
            "protocolSection",
            "resultsSection",
            "hasResults",
        ])

        params = {
            "pageSize": 1000,
            "filter.overallStatus": "COMPLETED",
            "countTotal": "true",
            "fields": fields,
        }

        term = query.get("term")
        condition = query.get("condition")
        intervention = query.get("intervention")

        if term is not None:
            params["query.term"] = term

        if condition is not None:
            params["query.cond"] = condition

        if intervention is not None:
            params["query.intr"] = intervention

        if patient_info is not None:
            params["query.patient"] = patient_info

        advanced_filters = []

        if studyType is not None:
            advanced_filters.append(f"AREA[StudyType]{studyType}")

        if excludeNctIds is not None:
            for nct_id in excludeNctIds:
                advanced_filters.append(f"NOT AREA[NCTId]{nct_id}")

        if advanced_filters:
            params["filter.advanced"] = " AND ".join(advanced_filters)

        studies = []
        total_count = 0
        url = "https://clinicaltrials.gov/api/v2/studies"

        while True:
            response = session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if total_count == 0:
                total_count = data.get("totalCount", 0)

            for study in data.get("studies", []):
                if hasResults and not study.get("hasResults", False):
                    continue

                studies.append(study)

            next_token = data.get("nextPageToken")
            if next_token is None:
                break

            params["pageToken"] = next_token

        paths = []

        if output_path is not None:
            paths = download_clinical_trials(studies, output_path)

        return {
            "query": query,
            "total_count": total_count,
            "studies": studies,
            "paths": paths,
        }

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(
            __search_query__,
            queries,
            repeat(patient_info),
            repeat(hasResults),
            repeat(studyType),
            repeat(excludeNctIds),
            repeat(output_path)
        )

    results = list(results)

    if excludeDuplicates:
        seen_nct_ids = set()

        for result in results:
            unique_studies = []

            for study in result["studies"]:
                nct_id = (
                    study.get("protocolSection", {})
                    .get("identificationModule", {})
                    .get("nctId")
                )

                if nct_id is None or nct_id in seen_nct_ids:
                    continue

                seen_nct_ids.add(nct_id)
                unique_studies.append(study)

            result["studies"] = unique_studies

    return results


if __name__ == "__main__":
    import time

    # Passing multiple queries ensures the thread pool actually has tasks to distribute concurrently
    test_queries = [
        {"term": "lung cancer", "condition": None, "intervention": None},
        {"term": "breast cancer", "condition": None, "intervention": None},
        {"term": "diabetes", "condition": None, "intervention": None},
        {"term": "hypertension", "condition": None, "intervention": None},
        {"term": "asthma", "condition": None, "intervention": None},
        {"term": "glaucoma", "condition": None, "intervention": None},
        {"term": "leukemia", "condition": None, "intervention": None},
        {"term": "melanoma", "condition": None, "intervention": None}
    ]

    # Test concurrency scaling
    worker_counts = [1, 2, 4, 8, 16]

    print(f"Benchmarking {len(test_queries)} concurrent queries...\n")
    print(f"{'Workers':<10} | {'Time (seconds)':<15} | {'Total Studies Retrieved'}")
    print("-" * 55)

    for workers in worker_counts:
        start_time = time.perf_counter()

        results = search_clinical_trials(
            queries=test_queries,
            hasResults=True,
            studyType='INTERVENTIONAL',
            num_workers=workers
        )

        elapsed_time = time.perf_counter() - start_time
        total_studies = sum(len(res["studies"]) for res in results)

        print(f"{workers:<10} | {elapsed_time:<15.4f} | {total_studies}")