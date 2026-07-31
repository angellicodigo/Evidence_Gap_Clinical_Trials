import requests
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
 
retry_strategy = Retry(
    total=8,                          # up to 8 retries per request
    backoff_factor=1,                 # sleeps 1s, 2s, 4s, 8s, 16s... between retries
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True,  # honor the server's Retry-After header if it sends one
)

max_connections = 64
session = requests.Session()
adapter = HTTPAdapter(
    max_retries=retry_strategy, 
    pool_connections=max_connections, 
    pool_maxsize=max_connections
)
session.mount("https://", adapter)
session.mount("http://", adapter)

def getClassByRxNormDrugId(
    rxcui: str, 
    classTypes: str | None = None, 
    relaSource: str | None = None, 
    ttys: list[str] | None = None, 
    ignoreNoneRela: bool = False
) -> list[dict[str, Any]]:
    url = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json"

    params = {"rxcui": rxcui}
    if classTypes is not None:
        params["classTypes"] = classTypes
    if relaSource is not None:
        params["relaSource"] = relaSource

    response = session.get(url, params=params)
    response.raise_for_status()

    results = response.json().get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])

    filtered = []
    for item in results:
        if ignoreNoneRela and not item.get("rela"):
            continue

        if ttys is not None:
            tty = item.get("minConcept", {}).get("tty")
            if tty not in ttys:
                continue

        filtered.append(item)

    return filtered

def getClassMembers(
    classId: str,
    className: str,
    classType: str,
    relaSource: str | None = None,
    rela: str | None = None,
    trans: int | None = None,
    ttys: list[str] | None = None,
    ignoreNoneRela: bool = False,
    extend: bool = False,
    extendClassTypes: list[str] | None = None,
) -> list[dict[str, Any]]:
    url = "https://rxnav.nlm.nih.gov/REST/rxclass/classMembers.json"

    results = []
    # Stores a set of tuples of (rxcui, classId)
    seen_drug_classes = set()
    # The basic set of drug members found if extend=Falseb
    base_rxcuis = set()
    # Keyed by rxcui — called at most once per drug across the entire function,
    # and reused by the extend block below so it never needs its own call.
    drug_classes_cache = {}

    if relaSource is None:
        rela_sources = getRelaSource(classType)
    else:
        rela_sources = [relaSource]

    extend_classTypes_str = " ".join(extendClassTypes) if extendClassTypes else None
    if extend and extend_classTypes_str is not None and classType not in extend_classTypes_str.split():
        extend_classTypes_str = f"{extend_classTypes_str} {classType}"

    # Calls the RxClass API
    def __fetch__(params: dict[str, Any]) -> list[dict[str, Any]]:
        response = session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("drugMemberGroup", {}).get("drugMember", [])

    def __drug_classes__(rxcui: str) -> list[dict[str, Any]]:
        if rxcui in drug_classes_cache:
            return drug_classes_cache[rxcui]
        
        classes = getClassByRxNormDrugId(
            rxcui=rxcui,
            classTypes=extend_classTypes_str,
            ttys=ttys,
            ignoreNoneRela=ignoreNoneRela,
        )

        if extend:
            # Broad call: covers this classType + extendClassTypes across all
            # relaSources. Used both to find this class's own rela and as the
            # extend lookup — so extend never needs a second call per drug.
            classes = getClassByRxNormDrugId(
                rxcui=rxcui,
                classTypes=extend_classTypes_str,
                ttys=ttys,
                ignoreNoneRela=ignoreNoneRela,
            )
        else:
            # Narrow call: only this classType. We don't care about anything
            # beyond this drug's relationship to classId.
            classes = getClassByRxNormDrugId(
                rxcui=rxcui,
                classTypes=classType,
            )

        drug_classes_cache[rxcui] = classes
        return classes

    # Formats the drug members pulled from RxClass API with additional info
    def __add_member__(member: dict[str, Any], current_rela_source: str) -> None:
        rxcui = member.get("minConcept", {}).get("rxcui")
        if not rxcui:
            return

        drug_class_tuple = (rxcui, classId)
        if drug_class_tuple in seen_drug_classes:
            return

        seen_drug_classes.add(drug_class_tuple)
        base_rxcuis.add(rxcui)

        # Single call per drug — narrow or broad depending on extend.
        # The result is cached so the extend block below gets it for free.
        classes = __drug_classes__(rxcui)
        match = next(
            (item for item in classes if item.get("rxclassMinConceptItem", {}).get("classId") == classId),
            None,
        )
        found_rela = match.get("rela") if match else None

        drug = dict(member)
        drug["classInfo"] = {
            "className": className,
            "classId": classId,
            "classType": classType,
            "rela": found_rela if ignoreNoneRela else None,
            "relaSource": current_rela_source,
        }
        results.append(drug)

    for current_rela_source in rela_sources:
        params = {
            "classId": classId, 
            "relaSource": current_rela_source
        }

        if trans is not None:
            params["trans"] = trans

        if ttys:
            params["ttys"] = " ".join(ttys)

        if rela is not None:
            params_copy = {
                **params, 
                "rela": rela
            }
            for member in __fetch__(params_copy):
                __add_member__(member, current_rela_source)
        else:
            for member in __fetch__(params):
                __add_member__(member, current_rela_source)

    # Extend logic: the cache is already fully populated above, no new
    # getClassByRxNormDrugId calls happen here.
    if extend and base_rxcuis:
        for rxcui in base_rxcuis:
            for item in drug_classes_cache.get(rxcui, []):
                item_rxcui = item.get("minConcept", {}).get("rxcui")
                c_id = item.get("rxclassMinConceptItem", {}).get("classId")

                drug_class_tuple = (item_rxcui, c_id)
                if drug_class_tuple in seen_drug_classes:
                    continue

                seen_drug_classes.add(drug_class_tuple)
                results.append({
                    "minConcept": item.get("minConcept", {}),
                    "classInfo": {
                        "className": item.get("rxclassMinConceptItem", {}).get("className"),
                        "classId": c_id,
                        "classType": item.get("rxclassMinConceptItem", {}).get("classType"),
                        "rela": item.get("rela"),
                        "relaSource": item.get("relaSource"),
                    }
                })

    return results


def getClassesMembers(
    classIds: list[str],
    classNames: list[str],
    classTypes: list[str],
    relaSources: list[str] | None = None,
    relas: list[str] | None = None,
    trans: int | None = None,
    ttys: list[str] | None = None,
    ignoreNoneRela: bool = False,
    extend: bool = False,
    extendClassTypes: list[str] | None = None,
    num_workers: int = 64,
) -> list[list[dict[str, Any]]]:

    if not (len(classIds) == len(classNames) == len(classTypes)):
        raise ValueError(
            "classIds, classNames, and classTypes must have the same length."
        )
    
    if relaSources is None:
        relaSources = [None] * len(classIds)
    
    if relas is None:
        relas = [None] * len(classIds)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                getClassMembers,
                classId,
                className,
                classType,
                relaSource,
                rela,
                trans,
                ttys,
                ignoreNoneRela,
                extend,
                extendClassTypes
            )
            for classId, className, classType, relaSource, rela in zip(classIds, classNames, classTypes, relaSources, relas)
        ]

        return [future.result() for future in futures]

def getRelaSource(classType: str | None = None) -> list[str] | dict[str, list[str]]:
    mapping = {
        'ATC1-4': ['ATCPROD', 'ATC'],
        'CHEM': ['DAILYMED', 'FDASPL', 'MEDRT'],
        'CVX': ['CDC'],
        'DISEASE': ['MEDRT'],
        'DISPOS': ['SNOMEDCT'],
        'EPC': ['DAILYMED', 'FDASPL'],
        'MOA': ['DAILYMED', 'FDASPL', 'MEDRT'],
        'PE': ['DAILYMED', 'FDASPL', 'MEDRT'],
        'PK': ['MEDRT'],
        'SCHEDULE': ['RXNORM'],
        'STRUCT': ['SNOMEDCT'],
        'TC': ['FMTSME'],
        'VA': ['VA']
    }
    return mapping[classType] if classType is not None else mapping


def getRelaDescription(relaSource: str | None = None, rela: str | None = None) -> str:
    descriptions = {  
        ("MEDRT", "may_treat"): "Drugs that may treat the disease.",
        ("MEDRT", "may_prevent"): "Drugs that may prevent the disease.",
        ("MEDRT", "CI_with"): "Drugs contraindicated for the disease.",
        ("MEDRT", "induces"): "Drugs that induce the pharmacokinetic property.",
        ("MEDRT", "has_MoA"): "Drugs with the specified mechanism of action.",
        ("MEDRT", "has_PE"): "Drugs with the specified physiologic effect.",
        ("MEDRT", "has_PK"): "Drugs with the specified pharmacokinetic property.",
        ("MEDRT", "site_of_metabolism"): "Drugs metabolized at the specified site.",
        ("MEDRT", "has_chemical_structure"): "Drugs with the specified chemical structure.",
        ("DAILYMED", "has_EPC"): "Drugs belonging to the Established Pharmacologic Class.",
        ("DAILYMED", "has_MoA"): "Drugs with the specified mechanism of action.",
        ("DAILYMED", "has_PE"): "Drugs with the specified physiologic effect.",
        ("FDASPL", "has_EPC"): "Drugs belonging to the Established Pharmacologic Class.",
        ("FDASPL", "has_MoA"): "Drugs with the specified mechanism of action.",
        ("FDASPL", "has_PE"): "Drugs with the specified physiologic effect.",
        ("ATC", "has_ATC"): "Drugs belonging to the ATC class.",
        ("ATCPROD", "has_ATC"): "Drug products belonging to the ATC class.",
        ("CDC", "has_CVX"): "Vaccines belonging to the CVX class.",
        ("RXNORM", "has_schedule"): "Controlled substances in the schedule.",
        ("SNOMEDCT", "has_disposition"): "Drugs with the specified disposition.",
        ("SNOMEDCT", "has_structure"): "Drugs with the specified structure.",
        ("VA", "has_VAClass"): "Drugs belonging to the VA drug class.",
        ("FMTSME", "has_TC"): "Drugs belonging to the therapeutic category.",
    }
    if relaSource is not None and rela is not None:
        return descriptions.get((relaSource, rela))
    
    return descriptions

def getClassTypeDescription(classType: str | None = None) -> str | dict[str, str] | None:
    descriptions = {
        "ATC1-4": """Anatomical Therapeutic Chemical (ATC) containing class levels 1 through 4. 
        The ATC classification system groups drugs according to the organ or system on which they act 
        and according to their chemical, pharmacological and therapeutic properties. The drugs are divided 
        into 14 main groups (first level), with two therapeutic/pharmacological subgroups (second and third levels). 
        The fourth level is a therapeutic/pharmacological/chemical subgroup and the fifth level is the chemical 
        substance. The second, third and fourth levels are often used to identify pharmacological subgroups when 
        these are considered to be more appropriate than therapeutic or chemical subgroups.""",

        "CHEM": """Chemical structure and classification containing chemicals or other drug ingredients, organized 
        into a chemical structure classification hierarchy. The high level non-MeSH classes, including the top level 
        node of "Substances and Cells", were created as parents of the MeSH classes contained in the tree, which are 
        part of the drugs and chemicals ("D" tree) category in MeSH. Note that some relations from MED-RT may not be 
        included if they map to classes not contained in the Chem tree.""",

        "CVX": """The CVX vaccine groupings from the CDC whose members are generic and branded drugs.""",

        "DISEASE": """Disease classification containing pathophysiologic as well as certain non-disease physiologic states 
        that are treated, prevented, or diagnosed by an ingredient or drug product. May also be used to describe contraindications. 
        The high level non-MeSH classes, including the top level node of "Diseases, Life Phases, Behavior Mechanisms and 
        Physiologic States", were created as parents of the MeSH classes contained in the tree. Specifically, the RxClass Disease 
        tree incorporates the following high-level MeSH classes (MeSH tree ids in parenthesis) and their subclasses:

            Diseases (C)
            Behavior and Behavior Mechanisms (F01)
            Mental Disorders (F03)
            Reproductive Physiological Phenomena (G08.686)
            Immune System Phenomena (G12)
            Age Groups (M01.060)

        Note that some MED-RT disease relations may not be included if they map to classes not contained in the tree.""",

        "DISPOS": """Drug classification from SNOMED CT based on the “dispositions” of medicinal products (e.g., mechanism of action).""",

        "EPC": """FDA Established Pharmacologic Classes. In support of the FDA Structured Product Labeling (SPL) initiative, 
        a hierarchy of FDA Established Pharmacologic Class (EPC) concepts are contained in MED-RT.""",

        "MOA": """Mechanism of Action in MED-RT, containing molecular, subcellular, or cellular effects of drug generic 
        ingredients, organized into a chemical function classification hierarchy, beneath the "Cellular or Molecular Interactions" 
        concept.""",

        "PE": """Physiologic Effects in MED-RT, containing tissue, organ, or organ system effects of drug generic ingredients, 
        organized into an organ system classification hierarchy, beneath the “Physiological Effects” concept.""",

        "PK": """Pharmacokinetics in MED-RT, containing collections of concepts describing the absorption, distribution, and 
        elimination of drug active ingredients, beneath the “Clinical Kinetics” concept.""",

        "SCHEDULE": """The Controlled Substances Act (CSA) drug schedules (1-5), whose members are drug products 
        (generic and branded).""",

        "STRUCT": """Drug classification from SNOMED CT based on the chemical structure of medicinal products.""",

        "TC": """Therapeutic Categories from MED-RT, a small, experimental collection of general therapeutic intents of drug 
        generic ingredients, organized into an organ system-oriented classification hierarchy, beneath the 
        "Therapeutic Categories [TC]" concept.""",

        "VA": """VA drug classes from VANDF, whose members are clinical drugs.""",
    }

    if classType is not None:
        return descriptions.get(classType.upper())

    return descriptions

if __name__ == '__main__':
    import time

    sample_class_ids = [
        "N0000175743",
        "N0000175739",
        "N0000175586",
        "N0000175658",
        "N0000175850",
        "N0000175629",
    ]

    sample_class_names = [
        "Melatonin Receptor Agonist",
        "Central Nervous System Stimulant",
        "Low Molecular Weight Heparin",
        "CD20-directed Radiotherapeutic Antibody",
        "Melanin Synthesis Inhibitors",
        "Increased Histamine Release",
    ]

    # All of these are EPC classes
    sample_class_types = ["EPC"] * len(sample_class_ids)

    # Use the same relaSource for every class
    sample_rela_sources = ["DAILYMED"] * len(sample_class_ids)

    worker_counts = [1, 2, 4, 10, 20, 32, 50]

    print(f"Benchmarking {len(sample_class_ids)} classes across worker counts...\n")
    print(f"{'Workers':<10} | {'Time (seconds)':<15} | {'Total Items Retrieved'}")
    print("-" * 55)

    for workers in worker_counts:
        start_time = time.perf_counter()

        results = getClassesMembers(
            classIds=sample_class_ids,
            classNames=sample_class_names,
            classTypes=sample_class_types,
            relaSources=sample_rela_sources,
            extend=True,
            num_workers=workers,
            ttys=["IN"],
            trans=1,
            ignoreNoneRela=True,
            extendClassTypes=["DISEASE", "MOA", "PE"]
        )

        elapsed_time = time.perf_counter() - start_time
        total_items = sum(len(res) for res in results)

        print(f"{workers:<10} | {elapsed_time:<15.4f} | {total_items}")
