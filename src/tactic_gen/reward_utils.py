


def reward_goals(initial_goals, final_goals):
    """
    Compare the initial goals with the final goals.
    """
    try:
        if (
            ((initial_goals is None or final_goals is None) 
                and initial_goals == final_goals)
            or ((initial_goals.goals is None or final_goals.goals is None) 
            and initial_goals.goals == final_goals.goals)
            or ((initial_goals.goals.goals is None or final_goals.goals.goals is None)
            and initial_goals.goals.goals == final_goals.goals.goals)
        ):  
            return 1
        elif final_goals is None or final_goals.goals is None:
            return 1
        initial_goals_ty = [goal.ty for goal in initial_goals.goals.goals]
        final_goals_ty = [goal.ty for goal in final_goals.goals.goals]
    except Exception as e:
        print("initial_goals", initial_goals)
        print("final_goals", final_goals)
        if initial_goals is not None:
            print("initial_goals.goals", initial_goals.goals)
        if final_goals is not None:
            print("final_goals.goals", final_goals.goals)
        raise e
        

    return -1 if initial_goals_ty == final_goals_ty else 1