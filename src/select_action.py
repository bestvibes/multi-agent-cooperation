from src.util_types import ActionCardinal

def select_action_chasee(action_type,
                            state: tuple,
                            chaser_next_action: ActionCardinal,
                            epsilon: float = 0.1) -> ActionCardinal:
    random_choice = True #binomial(1, epsilon)
    if random_choice:
        return action_type(random.randrange(0, len(action_type)))
    else:
        return chaser_next_action
