import numpy as np




def golden_model_rescale_up(
    data_in: int,
    input_zp_i: int,
    output_zp_i: int,
    shift_i: int,
    max_int_i: int,
    min_int_i: int,
    multiplier_i: int,
) -> int:
    """
    This function performs SIMD postprocessing of data given approximate algorithm of TOSA.rescale,
    with dynamically scaled shifts.
    """
    # Step 1: Subtract input zero point
    var_1 = data_in - input_zp_i

    # Step 2: Multiply with the multiplier avoiding overflow
    var_2 = np.int64(var_1) * np.int64(multiplier_i)

    # Step 3: Left shift one
    shifted_one = np.int64(
        1 << (shift_i - 1)
    )  # TODO: check if the minus one is actually correct

    # Step 4: Add shifted one
    var_3 = np.int64(var_2 + shifted_one)

    # Step 6: Shift right
    var_6 = np.int32(var_3 >> shift_i)

    # Step 7: Add output zero point
    var_7 = var_6 + np.int32(output_zp_i)

    # Step 8: Clip the values to be within min and max integer range
    var_8 = np.clip(var_7, min_int_i, max_int_i)

    return int(var_8)

golden_model_rescale_up(-113, 0,0, 10, 2 << 31 -1, -2 << 31, 1073741824)
