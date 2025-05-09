


def reward_goals(initial_goals, final_goals):
    """
    Compare the initial goals with the final goals.
    """
    if (
        (initial_goals is None or final_goals is None) 
        and initial_goals == final_goals
    ):
        return 1
    elif final_goals is None:
        return 1
    initial_goals_ty = [goal.ty for goal in initial_goals.goals.goals]
    final_goals_ty = [goal.ty for goal in final_goals.goals.goals]

    return -1 if initial_goals_ty == final_goals_ty else 1