"""
src/guidance.py
---------------
Static CBT therapy guidance lookup table.

One entry per distortion — no dataset required.
Each entry contains:
  - explanation  : plain-English description of what this distortion is
  - technique    : the CBT technique used to challenge it
  - challenge    : a Socratic question the therapist would ask
  - reframe      : an example of how to reframe the thought

Usage:
    from src.guidance import get_guidance
    g = get_guidance("Catastrophising/Magnification")
    print(g["challenge"])
"""

from src.config import DISTORTION_LABELS

# ── Lookup table ───────────────────────────────────────────────────────────────
_GUIDANCE: dict[str, dict] = {

    "All-or-nothing thinking": {
        "explanation": (
            "You see things in black-and-white, with no middle ground. "
            "If something isn't perfect, it's a total failure."
        ),
        "technique": "Continuum Method",
        "challenge": (
            "Is there a middle ground here? On a scale of 0–100, "
            "where does this situation actually fall?"
        ),
        "reframe": (
            "Instead of 'I failed completely', try: "
            "'I didn't get everything right, but I did succeed at parts of it.'"
        ),
    },

    "Overgeneralization": {
        "explanation": (
            "You draw a broad conclusion from a single event, "
            "using words like 'always', 'never', or 'everyone'."
        ),
        "technique": "Evidence Examination",
        "challenge": (
            "How many times has this actually happened? "
            "Are there any exceptions to this pattern you might be overlooking?"
        ),
        "reframe": (
            "Instead of 'I always mess things up', try: "
            "'This time didn't go well, but that doesn't define every situation.'"
        ),
    },

    "Mental filter": {
        "explanation": (
            "You focus exclusively on one negative detail and ignore "
            "the bigger picture, like a drop of ink that colours the whole glass."
        ),
        "technique": "Positive Data Log",
        "challenge": (
            "What else happened in this situation that you might be filtering out? "
            "If a friend described this to you, what would they notice?"
        ),
        "reframe": (
            "Instead of dwelling only on the one criticism, try: "
            "'Three people praised my work and one person had a concern — "
            "both pieces of feedback are real.'"
        ),
    },

    "Should statements": {
        "explanation": (
            "You hold rigid rules about how you or others must behave. "
            "'Should', 'must', and 'ought' create pressure and guilt when unmet."
        ),
        "technique": "Preference Reframing",
        "challenge": (
            "Where did this rule come from? Is it a realistic expectation, "
            "or a preference you've turned into a demand?"
        ),
        "reframe": (
            "Instead of 'I should never make mistakes', try: "
            "'I'd prefer to do well, and it's okay when I fall short sometimes.'"
        ),
    },

    "Labeling": {
        "explanation": (
            "You attach a global negative label to yourself or others "
            "based on a single event, going beyond the behaviour to the person."
        ),
        "technique": "Behaviour–Identity Separation",
        "challenge": (
            "Is one action the same as your entire identity? "
            "Would you define another person entirely by one mistake they made?"
        ),
        "reframe": (
            "Instead of 'I'm a loser', try: "
            "'I made a mistake in this situation — that's something I did, not who I am.'"
        ),
    },

    "Personalization": {
        "explanation": (
            "You blame yourself (or others) as the primary cause of events "
            "that are actually outside your full control."
        ),
        "technique": "Responsibility Pie Chart",
        "challenge": (
            "What other factors might have contributed to this outcome? "
            "If you listed every cause and gave each a percentage, "
            "how much is genuinely your responsibility?"
        ),
        "reframe": (
            "Instead of 'It's all my fault they're upset', try: "
            "'My actions were one factor — the situation had many contributing parts.'"
        ),
    },

    "Catastrophising/Magnification": {
        "explanation": (
            "You blow a problem out of proportion, imagining the worst possible "
            "outcome and treating it as likely or inevitable."
        ),
        "technique": "Decatastrophising",
        "challenge": (
            "What is the realistic best, worst, and most likely outcome here? "
            "Even if the worst happened, could you cope with it?"
        ),
        "reframe": (
            "Instead of 'This is a disaster and I'll never recover', try: "
            "'This is difficult and uncomfortable, but I've handled hard things before.'"
        ),
    },

    "Emotional Reasoning": {
        "explanation": (
            "You treat your feelings as facts. Because you feel something strongly, "
            "you assume it must be true."
        ),
        "technique": "Emotion vs. Fact Check",
        "challenge": (
            "What is the evidence for and against this thought, "
            "separate from how you're feeling right now? "
            "Feelings are real, but are they accurate information here?"
        ),
        "reframe": (
            "Instead of 'I feel stupid, so I must be stupid', try: "
            "'I feel embarrassed right now — that's an emotion, not a fact about my intelligence.'"
        ),
    },

    "Mind Reading": {
        "explanation": (
            "You assume you know what others are thinking, usually concluding "
            "they are thinking negatively about you — without checking."
        ),
        "technique": "Behavioural Experiment",
        "challenge": (
            "What actual evidence do you have for what they were thinking? "
            "Is there another explanation for their behaviour that you haven't considered?"
        ),
        "reframe": (
            "Instead of 'They didn't reply — they must be angry with me', try: "
            "'I don't know why they haven't replied yet. There could be many reasons.'"
        ),
    },

    "Fortune-telling": {
        "explanation": (
            "You predict that things will turn out badly and treat "
            "this prediction as a fact, even before anything has happened."
        ),
        "technique": "Probability Estimation",
        "challenge": (
            "What is the realistic probability that this will actually happen? "
            "Have you made similar predictions before that turned out to be wrong?"
        ),
        "reframe": (
            "Instead of 'I know I'm going to fail this', try: "
            "'I can't know the outcome yet. I'll prepare as well as I can and see.'"
        ),
    },
}


# ── Public API ─────────────────────────────────────────────────────────────────
def get_guidance(label: str) -> dict:
    """
    Return the guidance dict for a given distortion label.
    Raises KeyError if label is not found.

    Keys returned: explanation, technique, challenge, reframe
    """
    if label not in _GUIDANCE:
        raise KeyError(
            f"No guidance found for label: '{label}'. "
            f"Valid labels: {list(_GUIDANCE.keys())}"
        )
    return _GUIDANCE[label]


def all_labels_with_guidance() -> list[str]:
    """Return all distortion labels that have guidance entries."""
    return list(_GUIDANCE.keys())


# ── Sanity check on import ─────────────────────────────────────────────────────
_missing = [l for l in DISTORTION_LABELS if l not in _GUIDANCE]
if _missing:
    raise RuntimeError(
        f"guidance.py is missing entries for: {_missing}\n"
        "Add them to _GUIDANCE before running the API."
    )
