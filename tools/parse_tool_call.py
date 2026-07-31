from typing import Dict, Any
import re
import json
import ast

def parse_tool_call(text: str) -> Dict[str, Any]:
    params = {}
    pattern = re.compile(
        r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )
    
    matches = pattern.findall(text)
    
    if not matches:
        raise ValueError(f"""
            Failed to parse tool call. Expected XML tags '<parameter=...>'.\n
            Raw output:\n\n{text}
            """
        )

    for name, value in matches:
        value = value.strip()

        # Strip markdown backticks if the LLM wrapped the parameter content in ```
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value).strip()

        # Check for simple nulls/empty values
        if value in ["", "null", "None", "N/A"]:
            params[name] = None
        elif value.isdigit():
            params[name] = int(value)
        elif value.startswith("[") or value.startswith("{"):
            # Attempt strict JSON parsing first
            try:
                params[name] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass

            # Fallback 1: Parse Python-style syntax (handles None, True, False, single quotes)
            try:
                params[name] = ast.literal_eval(value)
                continue
            except (SyntaxError, ValueError):
                pass

            # Fallback 2: Replace Python keywords with JSON keywords and retry json.loads
            try:
                fixed_value = (
                    value.replace("None", "null")
                    .replace("True", "true")
                    .replace("False", "false")
                )
                params[name] = json.loads(fixed_value)
                continue
            except json.JSONDecodeError:
                params[name] = value
        else:
            params[name] = value

    return params