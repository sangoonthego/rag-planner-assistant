from langchain_core.tools import tool
from typing import Optional

@tool
def financial_analyzer(team_name: str, cost_limit_vnd: int) -> str:
    budget_data = {
        "Alpha Team": 145000000,
        "Meta Team": 85000000,
        "Beta Team": 210000000,
    }

    if team_name not in budget_data:
        return f"Not Found your Budget's Data '{team_name}'. Check again please!!!"
    
    actual_budget = budget_data[team_name]
    
    if actual_budget > cost_limit_vnd:
        delta = actual_budget - cost_limit_vnd
        return f"The Budget of {team_name} is {actual_budget:,} VNĐ, over {cost_limit_vnd:,} VNĐ. Over {delta:,} VNĐ."
    elif actual_budget <= cost_limit_vnd:
        return f"The Budget of {team_name} is {actual_budget:,} VNĐ, is around the limit of {cost_limit_vnd:,} VNĐ."
    else:
        return f"Cannot compare Data."


@tool
def check_schedule_conflict(task_name: str, date: str) -> str:
    if date == "2026-01-20":
        return "Conflict."
    else:
        return f"{date} available for '{task_name}'."