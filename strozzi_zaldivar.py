def detect_strozzi_zaldivar(
    df,
    time_col="Time",
    T_col="Tr_K",
    x_col=None,
    reactant_col="CA",
    reactant_initial=None,
):
    """_summary_

    Args:
        df (_type_): _description_
        time_col (str, optional): _description_. Defaults to "Time".
        T_col (str, optional): _description_. Defaults to "Tr_K".
        x_col (_type_, optional): _description_. Defaults to None.
        reactant_col (str, optional): _description_. Defaults to "CA".
        reactant_initial (_type_, optional): _description_. Defaults to None.

    Condition runaway:
        div = (d2X/dt2) / (dX/dt) + (d2T/dt2) / (dT/dt) > 0

        condition : div > 0
    """
