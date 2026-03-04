from orchestration.ai_orchestrator import AIOrchestrator

class ModelRouter:

    def __init__(self):

        self.orchestrator = AIOrchestrator()

    def predict(self, text):

        results = self.orchestrator.run_all_models(text)

        best_model = max(results, key=results.get)

        confidence = results[best_model]

        label = "Positive" if confidence > 0.5 else "Negative"

        return label, confidence, best_model