from typing import Any
import requests

def getRxConceptProperties(rxcui: str) -> dict[str, Any]:
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data.get("properties", {})

if __name__ == '__main__':
    rxcui = '9801'
    print(getRxConceptProperties(rxcui).get('name'))