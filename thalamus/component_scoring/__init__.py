"""Recommendation matrix builder: build scoring_matrix_*.json files for skill, memory, and tool routing."""

from .skills.composer import SkillMatrixComposer

__all__ = ["SkillMatrixComposer"]
