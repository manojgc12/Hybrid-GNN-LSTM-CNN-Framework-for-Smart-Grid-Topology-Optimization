def compare_losses(before, after):
    improvement = ((before - after) / before) * 100
    return improvement
