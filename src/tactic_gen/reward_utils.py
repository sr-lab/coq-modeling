def goals_exist(goals):
    return goals is not None and goals.goals is not None and goals.goals.goals is not None

def reward_goals(initial_goals, final_goals):
    """
    Compare the initial goals with the final goals.
    """
    if (not goals_exist(initial_goals) or goals_exist(initial_goals)) and not goals_exist(final_goals):
        return 1
    elif not goals_exist(initial_goals) and goals_exist(final_goals):
        print("issue")
        return -1
    else:
        initial_goals_ty = [goal.ty for goal in initial_goals.goals.goals]
        final_goals_ty = [goal.ty for goal in final_goals.goals.goals]
        return -1 if initial_goals_ty == final_goals_ty else 1