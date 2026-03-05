import re

from latex2sympy2_extended import NormalizationConfig, normalize_latex
from math_verify import LatexExtractionConfig, StringExtractionConfig, parse, verify


def parse_answer(text):
    parts = text.split("</think>")
    if len(parts) != 2:
        return None, None

    return parts[0].strip(), parts[1].strip()


def simple_additional_normalization(expr):
    percentage_pattern = r"^(\d+\.?\d*)(?:\\%|%)$"
    match_gt = re.fullmatch(percentage_pattern, expr)
    if match_gt:
        expr = match_gt.group(1)
    expr = expr.rstrip(".\\")
    return expr


def enhanced_additional_normalization(expr):
    percentage_pattern = r"^(\d+\.?\d*)(?:\\%|%)$"
    match_gt = re.fullmatch(percentage_pattern, expr)
    if match_gt:
        expr = match_gt.group(1)
    expr = expr.rstrip(".\\")
    expr = re.sub(r"\\ln\\b", r"\\log", expr)

    expr = expr.replace("°", r"^\\circ")

    def _deg_repl(m: re.Match) -> str:
        num = m.group("num")
        return f"({num}*\\pi/180)"

    expr = re.sub(
        r"(?P<num>[+-]?\d+(?:\.\d+)?)\s*(?:\^\s*\{\\circ\}|\^\s*\\circ|\\circ)\\b",
        _deg_repl,
        expr,
    )

    return expr


def math_equal(gt_answer, predicted_answer, _additional_normalization, take_modulo: int | None = None, **kwargs):
    if predicted_answer is None:
        return False

    gt_answer = str(gt_answer)
    predicted_answer = str(predicted_answer)

    verify_timeout = kwargs.pop("timeout_seconds", None)

    if take_modulo is not None:
        gt_answer = int(gt_answer) % take_modulo
        try:
            predicted_answer = int(predicted_answer) % take_modulo
        except Exception:
            predicted_answer = None
        return predicted_answer == gt_answer

    mcq_options = "ABCDEFGHIJ"
    norm_gt_mcq = gt_answer.strip()

    is_mcq = re.fullmatch("|".join(mcq_options), norm_gt_mcq)
    parsed_gt = parse(gt_answer, [StringExtractionConfig(strings=tuple(mcq_options))], parsing_timeout=None)
    parsed_pred = parse(predicted_answer, [StringExtractionConfig(strings=tuple(mcq_options))], parsing_timeout=None)
    if is_mcq:
        return verify(parsed_gt, parsed_pred, timeout_seconds=verify_timeout)

    gt_answer = _additional_normalization(gt_answer)
    predicted_answer = _additional_normalization(predicted_answer)

    normalized_gt = normalize_latex(gt_answer, NormalizationConfig)
    normalized_pred = normalize_latex(predicted_answer, NormalizationConfig)
    is_normalized_equal = normalized_gt.replace(" ", "") == normalized_pred.replace(" ", "")

    if is_normalized_equal:
        return True

    text_literal_pattern = r"[a-zA-Z ,]+"
    is_text_literal = re.fullmatch(text_literal_pattern, normalized_gt) and re.fullmatch(
        text_literal_pattern, normalized_pred
    )
    if is_text_literal:
        return False

    current_gt_answer = gt_answer
    current_predicted_answer = predicted_answer

    latex_env_search_pattern = r"\$.*\$|\\\(.*\\\)|\\\[.*\\\]|\\boxed\{"
    if not re.search(latex_env_search_pattern, current_gt_answer, re.DOTALL):
        current_gt_answer = f"${current_gt_answer}$"
    if not re.search(latex_env_search_pattern, current_predicted_answer, re.DOTALL):
        current_predicted_answer = f"${current_predicted_answer}$"

    parsed_gt = parse(current_gt_answer, [LatexExtractionConfig()], parsing_timeout=None)
    parsed_pred = parse(current_predicted_answer, [LatexExtractionConfig()], parsing_timeout=None)

    return verify(parsed_gt, parsed_pred, timeout_seconds=verify_timeout, **kwargs)


def search_regex(string: str, regex: str):
    match = re.findall(regex, string)
    if match:
        return match[-1]
    return None


def search_boxed(string: str):
    if "\\boxed" not in string:
        return None

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    if retval:
        left = "\\boxed{"
        try:
            assert retval[: len(left)] == left
            assert retval[-1] == "}"
            return retval[len(left) : -1]
        except AssertionError:
            return None

    return None


def extract_answer(string: str, extract_from_boxed: bool = True, extract_regex: str = r"The final answer is (.+)$", relaxed=False):
    if string is None:
        return None

    if relaxed:
        return search_regex(string, extract_regex) or search_boxed(string)

    if extract_from_boxed:
        return search_boxed(string)
    return search_regex(string, extract_regex)


def compute_reward(gt_answer, gen_trajectory):
    predicted_answer = extract_answer(gen_trajectory)
    if predicted_answer is None:
        return False

    gt_answer = str(gt_answer)
    predicted_answer = str(predicted_answer)
    result = math_equal(gt_answer, predicted_answer, simple_additional_normalization)
    if not result:
        replace_symbols = ["ln", "log", "°", "circ"]
        have_replace_symbols = False
        for symbols in replace_symbols:
            if symbols in gt_answer or symbols in predicted_answer:
                have_replace_symbols = True
                break
        if have_replace_symbols:
            result = math_equal(gt_answer, predicted_answer, enhanced_additional_normalization)

    return result
