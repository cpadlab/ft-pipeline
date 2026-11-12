"""
Paginated SLM discovery and parameter ceiling guardrail demonstration.

Demonstrates querying Hugging Face Hub via `SLMRegistry` with dynamic size
bounds, inspecting model metadata without downloading weights, and enforcing
upper parameter ceilings against oversized models.
"""
from __future__ import annotations

from ft.models import SLMRegistry


def main() -> None:
    """
    Demonstrates paginated SLM discovery and parameter ceiling validation.

    Initializes an `SLMRegistry` bounded at 4.5B parameters, retrieves the top
    instruction-tuned SLMs within `[1.0B, 3.5B]`, validates an eligible candidate
    model via lightweight Hub metadata inspection, and verifies guardrail
    rejection for an 8B model exceeding the ceiling.
    """
    registry = SLMRegistry(max_params_b=4.5)

    print("=== 1. HUGGING FACE HUB SLM CATALOG (Page 1) ===")
    page_1 = registry.search_models(
        query="Instruct",
        min_params_b=1.0,
        max_params_b=3.5,
        sort="downloads",
        page=1,
        page_size=5,
    )

    for item in page_1.items:
        print(
            f"- {item.source:<38} | {item.estimated_params_b:>4.2f}B | "
            f"arch={item.model_type:<8} | downloads={item.downloads:>9,} | gated={item.gated}"
        )
    print(f"More pages available?: {page_1.has_more}\n")

    print("=== 2. SLM GUARDRAIL VALIDATION ===")
    valid_profile = registry.validate_model("google/gemma-2-2b-it")
    print(f"[OK] Valid model: {valid_profile.source} (~{valid_profile.estimated_params_b}B)\n")

    try:
        registry.validate_model("meta-llama/Llama-3.1-8B-Instruct")
    except ValueError as exc:
        print(f"[BLOCKED] Guardrail triggered: {exc}")


if __name__ == "__main__":
    main()