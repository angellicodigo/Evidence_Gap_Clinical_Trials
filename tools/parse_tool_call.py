from typing import Dict, Any
import re
import json

def parse_tool_call(text: str) -> Dict[str, Any]:
    params = {}
    pattern = re.compile(
        r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )

    for name, value in pattern.findall(text):
        value = value.strip()
        if value in ["", "null", "None", "N/A"]:
            value = None
        elif value.isdigit():
            value = int(value)
        elif value.startswith("["):
            value = json.loads(value)

        params[name] = value

    return params

if __name__ == "__main__":
    test = """
        </think>
        <tool_call>
        <function=search_clinical_trials>
        <parameter=terms>
        ["ravulizumab", "Ultomiris", "anti-C5", "complement inhibitor", "ravulizumab", "Ultomiris", "ravulizumab", "Ultomiris", "ravulizumab", "Ultomiris", "ravulizumab", "Ultomiris", "ravulizumab", "Ultomiris", null, null, null, null, "ravulizumab", "Ultomiris", "ravulizumab", "Ultomiris", "ravulizumab", "Ultomiris", "ravulizumab", "Ultomiris", "ravulizumab", "Ultomiris", null, null, null, null]
        </parameter>
        <parameter=conditions>
        ["Paroxysmal Hemoglobinuria", "Paroxysmal Hemoglobinuria", "Paroxysmal Hemoglobinuria", "Paroxysmal Hemoglobinuria", "PNH", "PNH", "atypical hemolytic uremic syndrome", "aHUS", "Myasthenia Gravis", "Neuromyelitis Optica Spectrum Disorder", "Hemoglobinuria, Paroxysmal", "Hemoglobinuria, Paroxysmal", null, null, "Paroxysmal Hemoglobinuria", "PNH", "atypical hemolytic uremic syndrome", "aHUS", "Paroxysmal Hemoglobinuria", "Paroxysmal Hemoglobinuria", "PNH", "PNH", "atypical hemolytic uremic syndrome", "aHUS", "Myasthenia Gravis", "Neuromyelitis Optica Spectrum Disorder", "Hemoglobinuria, Paroxysmal", "Hemoglobinuria, Paroxysmal", "Paroxysmal Hemoglobinuria", "PNH", "atypical hemolytic uremic syndrome", "aHUS"]
        </parameter>
        <parameter=interventions>
        ["ravulizumab", "Ultomiris", "ravulizumab", "ravulizumab", "ravulizumab", "Ultomiris", "ravulizumab", "Ultomiris", "ravulizumab", "Ultomiris", "ravulizumab", "Ultomiris", "ravulizumab", "Ultomiris", "ravulizumab", "Ultomiris", "ravulizumab", "ravulizumab", "ravulizumab", "ravulizumab", null, null, null, null, null, null, null, null, null, null, "Ultomiris", "Ultomiris", "Ultomiris", "Ultomiris"]
        </parameter>
        </function>
        </tool_call>
    """

    result = parse_tool_call(test)
    print(len(result['terms']))
    print(len(result['conditions']))
    print(len(result['interventions']))