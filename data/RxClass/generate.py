import networkx as nx
from pathlib import Path
from typing import Any
import sys
import requests
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.RxClass import getRelaSource, session

def hasClassMembers(
    classId: str,
    classType: str,
    relaSource: str | None = None,
    trans: int | None = None,
    ttys: list[str] | None = None,
    ignoreNoneRela: bool = False,
    max_retries: int = 8,
    backoff_factor: float = 2.0,
    inter_request_delay: float = 0.25,
) -> bool:
    url = "https://rxnav.nlm.nih.gov/REST/rxclass/classMembers.json"
    rela_sources = [relaSource] if relaSource is not None else getRelaSource(classType)

    def __check__(params: dict[str, Any]) -> bool:
        for attempt in range(max_retries):
            try:
                time.sleep(inter_request_delay)
                response = session.get(url, params=params)
                response.raise_for_status()
                members = response.json().get("drugMemberGroup", {}).get("drugMember", [])

                if not members:
                    return False

                if not ignoreNoneRela:
                    return True

                # Only count members that actually have a named rela
                return any(member.get("rela") for member in members)

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 429:
                    retry_after = e.response.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else backoff_factor * (2 ** attempt)
                    print(f"[hasClassMembers] 429 for classId={classId} — waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    if attempt == max_retries - 1:
                        print(f"[hasClassMembers] Exhausted retries for classId={classId}, treating as empty.")
                        return False
                else:
                    print(f"[hasClassMembers] HTTP {status} for classId={classId} — skipping.")
                    return False

            except requests.exceptions.RequestException as e:
                wait = backoff_factor * (2 ** attempt)
                print(f"[hasClassMembers] Request error for classId={classId}: {e} — retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    print(f"[hasClassMembers] Exhausted retries for classId={classId}, treating as empty.")
                    return False

        return False

    for current_rela_source in rela_sources:
        params: dict[str, Any] = {"classId": classId, "relaSource": current_rela_source}

        if trans is not None:
            params["trans"] = trans

        if ttys:
            params["ttys"] = " ".join(ttys)

        if __check__(params):
            return True

    return False

def getClassTree(classId: str, classType: str | None = None) -> list[dict[str, Any]]: 
    url = "https://rxnav.nlm.nih.gov/REST/rxclass/classTree.json"
    params = {"classId": classId}
    if classType is not None:
        params['classType'] = classType

    response = session.get(url, params=params)
    response.raise_for_status()
    return response.json().get("rxclassTree", [])


def classTreeToGraph(tree: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()

    def dfs(node: dict):
        concept = node["rxclassMinConceptItem"]
        classId = concept["classId"]
        G.add_node(classId, **concept)

        for child in node.get("rxclassTree", []):
            childConcept = child["rxclassMinConceptItem"]
            G.add_node(childConcept["classId"], **childConcept)
            G.add_edge(classId, childConcept["classId"])
            dfs(child)

    for root in tree:
        dfs(root)

    return G


def getLeaves(classId: str, classType: str) -> list[dict[str, Any]]:
    tree = getClassTree(classId, classType)
    graph = classTreeToGraph(tree)

    leaves = []
    for node in graph.nodes:
        if graph.out_degree(node) == 0:
            leaf = dict(graph.nodes[node])
            leaves.append(leaf)

    return leaves


def findClassByName(className: str, classTypes: str | None = None) -> dict[str, str]:
    if className == 'Disease': 
        url = "https://rxnav.nlm.nih.gov/REST/rxclass/classContext.json"
        res = session.get(url, params={"classId": "X3"})
        res.raise_for_status()
        return res.json().get("classPathList", {}).get("classPath", [])[0]["rxclassMinConcept"][0]

    url = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byName.json"

    params = {
        "className": className
    }

    if classTypes is not None:
        params['classTypes'] = classTypes

    response = session.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return data.get("rxclassMinConceptList", {}).get("rxclassMinConcept", [])[0]


def generate_files(classId: str, classType: str, output_path: Path, ignoreEmptyClasses: bool = False) -> None:
 
    leaves = getLeaves(classId, classType)
    seen = set()
 
    for leaf in leaves:
        name = leaf.get("className")
        cid = leaf.get("classId")
        if not name or not cid:
            continue
 
        if ignoreEmptyClasses:
            if not hasClassMembers(
                classId=cid,
                classType=classType,
                ttys=["IN"],
                trans=1,
            ):
                continue
 
        seen.add((name, cid))
 
    output_path.mkdir(parents=True, exist_ok=True)
    outfile = output_path / f"{classType}.txt"
    
    with outfile.open("w", encoding="utf-8") as f:
        for name, cid in sorted(seen):
            f.write(f"{name}\t{cid}\n")


if __name__ == '__main__':
    class_configs = [
        ("MOA", "Mechanism of Action"),
        ("PE", "Physiologic Effects"),
        ("DISEASE", "Disease"),
    ]

    path = Path("/sc/arion/projects/EHR_ML/lia38/Evidence_Gap_Clinical_Trials/data/RxClass")

    for classType, className in class_configs:
        result = findClassByName(className, classType)

        # Generate the text file, ignoring empty classes if desired
        generate_files(
            classId=result.get('classId'),
            classType=classType,
            output_path=path,
            ignoreEmptyClasses=True
        )