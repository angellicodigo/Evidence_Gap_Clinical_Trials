import requests
from typing import Any
import random
import networkx as nx
import pandas as pd
from pathlib import Path


def getAllClasses(classTypes: str) -> list[dict[str, str]]:
    url = "https://rxnav.nlm.nih.gov/REST/rxclass/allClasses.json"
    params = {
        "classTypes": classTypes
    }
    response = requests.get(url, params=params)
    response.raise_for_status() 
    data = response.json()
    return data.get("rxclassMinConceptList", {}).get("rxclassMinConcept", [])


def getRelas(relaSource: str):
    url = "https://rxnav.nlm.nih.gov/REST/rxclass/relas.json"
    params = {
        "relaSource": relaSource
    }
    response = requests.get(url, params=params)
    response.raise_for_status() 
    data = response.json()
    return data.get("relaList", {}).get("rela", [])


def getClassMembers(classId: str, relaSource: str, rela: str | None = None, trans: int | None = None, ttys: list[str] | None = None, ignoreNoneRela: bool = False) -> dict[str, list[dict[str, Any]]]:
    url = "https://rxnav.nlm.nih.gov/REST/rxclass/classMembers.json"

    params = {
        "classId": classId,
        "relaSource": relaSource,
    }

    if trans is not None:
        params["trans"] = trans
    if ttys:
        params["ttys"] = " ".join(ttys)

    result = {}

    def __fetch__():
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get("drugMemberGroup", {}).get("drugMember", [])

    if rela is not None:
        params["rela"] = rela
        result[rela] = __fetch__()

    else:
        # First try without specifying a rela
        members = __fetch__()
        if members and not ignoreNoneRela:
            result["None"] = members # Which actually does work
        else:
            # Then try all relationships based on relaSource
            for rela in getRelas(relaSource):
                params["rela"] = rela
                members = __fetch__()
                if members:
                    result[rela] = members

    # If empty, it is likely that the class has no members
    return result

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

    if classType is not None:
        return mapping[classType]

    return mapping

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


def getClassTree(classId: str, classType: str | None = None) -> list[dict[str, Any]]: 
    url = "https://rxnav.nlm.nih.gov/REST/rxclass/classTree.json"
    params = {
        "classId": classId,
    }

    if classType is not None:
        params['classType'] = classType

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    # Returns a list of dicts where each one is a root of a path
    return data.get("rxclassTree", [])

def getClassContexts(classId: str) -> list[list[dict[str, str]]]: 
    url = "https://rxnav.nlm.nih.gov/REST/rxclass/classContext.json"
    params = {
        "classId": classId,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return [
        path["rxclassMinConcept"]
        for path in data.get("classPathList", {}).get("classPath", [])
    ]

def classTreeToGraph(tree: list[dict]) -> nx.DiGraph:
    # Directed graph
    G = nx.DiGraph()

    def dfs(node: dict):
        concept = node["rxclassMinConceptItem"]
        classId = concept["classId"]

        # The classId is the identifier of the node
        # while values in concept are attributes
        G.add_node(classId, **concept)

        for child in node.get("rxclassTree", []):
            childConcept = child["rxclassMinConceptItem"]

            G.add_node(childConcept["classId"], **childConcept)
            G.add_edge(classId, childConcept["classId"])

            dfs(child)

    for root in tree:
        dfs(root)

    return G

def getLeaves(graph: nx.DiGraph) -> list[dict[str, str]]:
    leaves = []

    for node in graph.nodes:
        if graph.out_degree(node) == 0:
            leaves.append(dict(graph.nodes[node]))

    return leaves

def buildClassDrugRela(classId: str, classType: str, output_path: Path | None = None, trans: int | None = None, ttys: list[str] | None = None, ignoreNoneRela: bool = False):
    columns = ["Type", "RXCUI", "RxNorm Name", "Relation", "Relation Source", "Class Relation"]

    tree = getClassTree(classId, classType)
    graph = classTreeToGraph(tree)
    classes = getLeaves(graph)
    relaSources = getRelaSource(classType)

    dfs = {}

    for class_dict in classes:
        current_class_id = class_dict["classId"]
        class_name = class_dict["className"]

        rows = []

        for relaSource in relaSources:
            members = getClassMembers(
                current_class_id,
                relaSource,
                trans=trans,
                ttys=ttys,
                ignoreNoneRela=ignoreNoneRela,
            )

            if not members:
                continue

            for rela, drugs in members.items():
                for drug in drugs:
                    concept = drug.get("minConcept", {})
                    nodeAttr = drug.get("nodeAttr", [])
                    if concept is None or nodeAttr is None:
                        continue 
                    
                    rows.append(
                        {
                            "Type": concept.get("tty"), 
                            "RXCUI": concept.get("rxcui"), 
                            "RxNorm Name": concept.get("name"), 
                            "Relation": nodeAttr[2]['attrValue'], 
                            "Relation Source": relaSource,
                            "Class Relation": rela
                        }
                    )
        if not rows:
            continue

        df = pd.DataFrame(rows, columns=columns)
        dfs[class_name] = df

        if output_path is not None:
            output_dir = output_path / classType
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{class_name}.parquet"
            df.to_parquet(output_file, index=False)

    return dfs

if __name__ == '__main__':
    classId = 'X3'
    classType = 'DISEASE'
    

    # path = Path("/sc/arion/projects/EHR_ML/lia38/nemotronL40/data/responses")
    # print(buildClassDrugRela(classId, classType, output_path=path, trans=1, ttys=["IN"], ignoreNoneRela=True))
    