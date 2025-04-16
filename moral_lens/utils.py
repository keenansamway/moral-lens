
from dataclasses import dataclass
import re
from typing import List, Optional, Tuple, Dict
import yaml
from IPython.display import display
from IPython.display import HTML

from moral_lens.config import ModelConfig


def mydisplay(df, height=1000):
    display(HTML(
        f"<div style='height: {height}px; overflow: auto; width: fit-content'>" + df.to_html() + "</div>"
    ))


def load_yaml_file(path: str) -> Dict:
    """Load a YAML file and return its contents."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_goup_name_variations(value: str) -> List[str]:
    """
    Generate variations of a given value for matching in text.
    This function generates variations of a given value by:
        - Returning the original value.
        - Replacing any digits in the value with their word equivalents (e.g., '1' -> 'one').
        - Adding articles ('a' or 'an') for singular items.

    Args:
        value (str): The original value to generate variations for.

    Returns:
        List[str]: A list of variations for the given value.
    """
    value = value.lower()
    variations = [value]

    # Number mapping dictionary
    num2str = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    str2num = {v: k for k, v in num2str.items()}

    words = value.split()
    if words and len(words) > 1:
        first_word = words[0]
        rest = ' '.join(words[1:])

        # Add 'a' or 'an'; '1'; and 'one' prefixes
        if first_word in ('1', 'one', 'a', 'an'):
            article = 'an' if rest[0].lower() in 'aeiou' else 'a'
            variations.append(f"{article} {rest}")
            variations.append(f"1 {rest}")
            variations.append(f"one {rest}")

        # Add string prefix
        elif first_word in num2str.keys():
            variations.append(f"{num2str.get(first_word, first_word)} {rest}")

        # Add digit prefix
        elif first_word in str2num.keys():
            variations.append(f"{str2num.get(first_word, first_word)} {rest}")

        # Add 'the' prefix
        variations.append(f"the {rest}")

        # Add the rest without a prefix
        # variations.append(f"{rest}")  # breaks when testing for 'men' in 'women' & 'man' in 'woman'

    return list(dict.fromkeys(variations))  # Remove duplicates while preserving order



def parse_keyword_text(text: str, keyword: str) -> str:
    """
    Find the keyword from the text.

    Args:
        text (str): The text to search for the keyword.
        keyword (str): The keyword to look for in the text.

    Returns:
        str: The extracted keyword text or an empty string if none found.

    Matches formats such as:
    - `<keyword>...</keyword>`
    - `keyword: ...\nanotherword:`
    - `keyword: ...`

    For example, keywords can be: `reasoning`, `decision`, `answer`, etc.
    """
    text = text.replace("*", "") # Clean up any stray asterisks that might be in the text

    text = text.replace("_", " ") # Replace underscores with spaces
    keyword = keyword.replace("_", " ") # Replace underscores with spaces

    # Check for `<keyword>...</keyword>` format
    keyword_match = re.search(
        rf"<{keyword}>(.*?)</{keyword}>",
        text, re.IGNORECASE | re.DOTALL
    )
    if keyword_match:
        keyword_text = keyword_match.group(1).strip()
        return keyword_text

    # Check for `<keyword>...<anotherword>` format
    keyword_match = re.search(
        rf"<{keyword}>\s*((?:(?!<\w+>).|\n)*)",
        text, re.IGNORECASE | re.DOTALL | re.MULTILINE
    )
    if keyword_match:
        keyword_text = keyword_match.group(1).strip()
        return keyword_text

    # Check for `keyword: ...\nanotherword:` or `keyword: ...` format
    keyword_match = re.search(
        # rf"{keyword}:\s*((?:(?!^[a-z0-9_ ]+?:).)*)",
        rf"^{keyword}:\s*(.+?)(?:\n[a-zA-Z0-9_ ]+?:|\Z)",
        # rf"{keyword}:\s*(.+?)(?=\n(?:Answer \d+:|Decision:)|\Z)",
        text, re.IGNORECASE | re.DOTALL | re.MULTILINE
    )
    if keyword_match:
        keyword_text = keyword_match.group(1).strip()
        return keyword_text

    # If no match found, return an empty string
    return ""


def parse_keyword_text_(text: str, keyword: str, endword: str = None) -> str:
    text = text.replace("*", "")  # Clean up any stray asterisks that might be in the text
    text = text.replace("_", " ")  # Replace underscores with spaces
    keyword = keyword.replace("_", " ")  # Replace underscores with spaces

    # Ensure keyword and endword are at the beginning of a line
    if endword is not None and endword != "":
        keyword_match = re.search(
            rf"(?<={keyword}:)(.*?)(?={endword}:|$)",
            text, re.IGNORECASE | re.DOTALL
        )
    else:
        keyword_match = re.search(
            rf"(?<={keyword}:)(.*)",
            text, re.IGNORECASE | re.DOTALL | re.MULTILINE
        )
    if keyword_match:
        return keyword_match.group(1).strip()

    # If no match found, return an empty string
    return ""

def parse_reasoning_and_decision(text: Optional[str]) -> Optional[Tuple[str, str]]:
    """
    Parse the reasoning and decision from the text.
    """
    if text is None:
        return None, None

    # reasoning = parse_keyword_text(text, "reasoning")
    # decision = parse_keyword_text(text, "decision")

    reasoning = parse_keyword_text_(text, "reasoning", "decision")
    decision = parse_keyword_text_(text, "decision")
    return reasoning, decision


def fuzzy_match_decisions(value: str, possible_values: List[str]) -> Optional[str]:
    """
    Fuzzy match the decision in the text against a list of possible values.
    """
    # value = value.strip()
    # # Create a mapping from variations to their original values
    # variation_map = {}
    # for possible_value in possible_values:
    #     variations = get_goup_name_variations(possible_value)
    #     for variation in variations:
    #         variation_map[variation] = possible_value

    # # Direct lookup
    # return variation_map.get(value.lower(), value)

    value = value.strip().lower()

    contains_pv1 = False
    contains_pv2 = False

    pv1 = possible_values[0].lower()
    pv2 = possible_values[1].lower()

    for variation in get_goup_name_variations(pv1):
        if variation in value:
            contains_pv1 = True
            break

    for variation in get_goup_name_variations(pv2):
        if variation in value:
            contains_pv2 = True
            break

    if contains_pv1 and contains_pv2:
        # If both variations are found, return an empty string
        return ""
    elif contains_pv1:
        # If only pv1 is found, return pv1
        return pv1
    elif contains_pv2:
        # If only pv2 is found, return pv2
        return pv2
    else:
        # If neither variation is found, return an empty string
        return ""
