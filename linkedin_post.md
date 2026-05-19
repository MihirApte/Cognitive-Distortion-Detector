# LinkedIn Post Draft

---

I built a tool that reads your thoughts and flags cognitive distortions from CBT psychology.

Type something like "I always mess everything up" — it detects the distortion (Overgeneralization), shows you which words triggered it, and gives you the CBT technique to challenge it.

Under the hood:
- Fine-tuned DistilBERT on 2,500+ annotated therapist Q&A examples
- Multi-label classification across 10 distortion types (Beck's taxonomy)
- LIME for word-level explainability — so you can see *why* the model flagged something
- FastAPI backend + vanilla JS frontend, deployed on HuggingFace Spaces

This started as a conversation with [Psychology Collaborator's Name] about how often people dismiss their own distorted thinking because it *feels* true. That framing shaped the whole design — the app doesn't just classify, it gives you the Socratic question a therapist would actually ask.

The F1 scores are modest (macro ~0.23), and I'm being transparent about that. The dataset is small, the labels overlap heavily, and the task is genuinely hard — even human annotators disagree. But the model captures meaningful signal, and the LIME explanations make the predictions interpretable rather than just a black box.

Full write-up, notebooks, and code on GitHub. Live demo on HuggingFace Spaces (10–15s response time on CPU — free tier, so worth the wait).

→ Demo: [HuggingFace Spaces link]
→ Code: [GitHub link]

#NLP #MachineLearning #CBT #MentalHealth #Python #HuggingFace #PortfolioProject

---

**Notes before posting:**
- Replace [Psychology Collaborator's Name] with their actual name (get their permission first)
- Add the actual HuggingFace Spaces URL once deployed
- Add the GitHub repo URL
- Tag the collaborator if they're on LinkedIn and have agreed to it
- Optionally attach a screenshot of the UI for higher engagement
