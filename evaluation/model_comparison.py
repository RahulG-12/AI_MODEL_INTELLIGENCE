def compare_models(results):

    best_model = None
    best_score = 0

    for model, metrics in results.items():

        if metrics["accuracy"] > best_score:
            best_score = metrics["accuracy"]
            best_model = model

    return best_model