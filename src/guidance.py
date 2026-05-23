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
  - lime_hint    : one-line explanation of what kinds of words signal this distortion

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
            "Your mind is framing this as all-or-nothing — either perfect or a complete failure, "
            "with nothing in between. This is one of the most common thinking traps, especially "
            "under pressure, but it's almost never an accurate picture of reality."
        ),
        "technique": "Continuum Method",
        "challenge": (
            "If you put this situation on a scale of 0 (total failure) to 100 (perfect success), "
            "where does it honestly land? What would a score of 60 look like?"
        ),
        "reframe": (
            "Try replacing the all-or-nothing version with: "
            "'This didn't go the way I wanted — but that's not the same as it being a complete failure. "
            "There were parts I handled, and parts to improve.'"
        ),
        "lime_hint": (
            "Absolute words like \"completely\", \"total\", \"perfect\", \"never\", or \"failure\" "
            "signal black-and-white thinking that leaves no room for nuance."
        ),
    },

    "Overgeneralization": {
        "explanation": (
            "You're taking one event — or a few — and drawing a sweeping conclusion that applies "
            "to everything. Words like 'always' and 'never' are a giveaway. One difficult experience "
            "doesn't define a permanent pattern."
        ),
        "technique": "Evidence Examination",
        "challenge": (
            "How many times has this actually happened? Can you think of even one exception — "
            "a time when things went differently? What does the full picture look like, "
            "not just the recent example?"
        ),
        "reframe": (
            "Try replacing the overgeneralized version with: "
            "'This particular situation didn't go well. That doesn't mean it always goes this way — "
            "this is one data point, not a life sentence.'"
        ),
        "lime_hint": (
            "Words like \"always\", \"never\", \"every time\", \"everyone\", or \"no one\" "
            "signal a broad conclusion drawn from limited evidence."
        ),
    },

    "Mental filter": {
        "explanation": (
            "You're zooming in on one negative detail and letting it colour your entire view of "
            "the situation — filtering out anything positive or neutral. It's like a single drop "
            "of ink that makes a whole glass of water look dark."
        ),
        "technique": "Positive Data Log",
        "challenge": (
            "What else happened in this situation that you might be overlooking? "
            "If someone who cared about you was watching, what would they have noticed "
            "alongside the part you're focused on?"
        ),
        "reframe": (
            "Try widening the lens: "
            "'Yes, that one thing went wrong — and other things went okay. "
            "Both are real parts of what happened, and the negative detail doesn't cancel the rest.'"
        ),
        "lime_hint": (
            "Words that zero in on a single flaw or negative moment — while the surrounding "
            "context is ignored — are the signature of a mental filter at work."
        ),
    },

    "Should statements": {
        "explanation": (
            "You're holding yourself (or someone else) to a rigid rule — something that 'must', "
            "'should', or 'ought' to be a certain way. These rules often come from old expectations "
            "or external pressures, and when they're not met, the result is guilt, shame, or frustration."
        ),
        "technique": "Preference Reframing",
        "challenge": (
            "Where did this rule come from? Is it genuinely realistic, or is it a preference "
            "you've turned into a demand? What would happen if you replaced 'I should' "
            "with 'I'd prefer to' — how does that change how it feels?"
        ),
        "reframe": (
            "Try softening the demand: "
            "'I'd prefer to handle this without struggling, and it's okay that I sometimes don't. "
            "That makes me human, not inadequate.'"
        ),
        "lime_hint": (
            "Words like \"should\", \"must\", \"ought to\", \"have to\", or \"need to\" "
            "signal rigid internal rules that create pressure when unmet."
        ),
    },

    "Labeling": {
        "explanation": (
            "You've taken one action or mistake and used it to define yourself (or someone else) "
            "as a whole person. A label like 'I'm a failure' is an identity claim — it goes "
            "far beyond what actually happened and treats a behaviour as the entire person."
        ),
        "technique": "Behaviour–Identity Separation",
        "challenge": (
            "Would you define a close friend entirely by one mistake they made? "
            "What's the difference between 'I did something wrong' and 'I am wrong as a person'? "
            "Is one action really the full story of who you are?"
        ),
        "reframe": (
            "Try separating the action from the identity: "
            "'I made a mistake in this situation. That's something I did — not a verdict "
            "on who I am. People who care about quality sometimes fall short of it.'"
        ),
        "lime_hint": (
            "Global identity words — \"loser\", \"failure\", \"idiot\", \"worthless\" — "
            "attached to a whole person rather than a specific behaviour signal labeling."
        ),
    },

    "Personalization": {
        "explanation": (
            "You're taking personal responsibility for something that had many contributing factors — "
            "as if you were the sole cause. This usually leads to excessive guilt or blame "
            "for outcomes that were never fully in your control."
        ),
        "technique": "Responsibility Pie Chart",
        "challenge": (
            "If you listed every factor that contributed to this outcome and gave each a percentage, "
            "how much is genuinely yours? What other people, circumstances, or variables "
            "played a role that had nothing to do with you?"
        ),
        "reframe": (
            "Try distributing responsibility more accurately: "
            "'My actions were one part of this — but there were other factors involved. "
            "I can own my share without carrying the whole weight.'"
        ),
        "lime_hint": (
            "Words like \"my fault\", \"because of me\", \"I caused\", or \"if only I had\" "
            "signal taking on more personal responsibility than the situation warrants."
        ),
    },

    "Catastrophising/Magnification": {
        "explanation": (
            "You're imagining the worst possible outcome and treating it as though it's likely "
            "or inevitable. The situation may be genuinely difficult — but your mind is "
            "magnifying it far beyond what the evidence supports."
        ),
        "technique": "Decatastrophising",
        "challenge": (
            "What's the realistic best, worst, and most likely outcome here — honestly? "
            "If the worst did happen, have you coped with hard things before? "
            "What does your track record actually say about your ability to handle difficulty?"
        ),
        "reframe": (
            "Try grounding it: "
            "'This is genuinely difficult, and I don't lik