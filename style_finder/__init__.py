"""Style Finder: multimodal RAG for fashion image analysis."""

__all__ = ["StyleFinder", "Settings"]


def __getattr__(name):
    # Lazy imports keep `import style_finder` cheap (no torch/watsonx load)
    # until the pipeline is actually needed.
    if name == "StyleFinder":
        from .pipeline import StyleFinder

        return StyleFinder
    if name == "Settings":
        from .config import Settings

        return Settings
    raise AttributeError(name)
