


def compare_goals(initial_goals, final_goals):
    """
    Compare the initial goals with the final goals.
    """
    initial_goals_ty = [goal.ty for goal in initial_goals]
    final_goals_ty = [goal.ty for goal in final_goals]
    return initial_goals_ty == final_goals_ty