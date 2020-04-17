def _findExponent(in_val, start=2):
    i = 0
    val = 2**i
    while val < in_val:
        i += 1
        val = start**i
    return i
