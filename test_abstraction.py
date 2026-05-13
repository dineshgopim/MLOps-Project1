from pipeline import PipelineStep

print(f"---Testing Abstraction---")

try:        # testing whether we can create objects directly in ABC
    invalid_step = PipelineStep()
except TypeError as e:
    print(f"Successfully Caught the expected error: {e}")

class BrokenLoader(PipelineStep):       # testing whether we can create objects without 'exucte' method since inheriting from parent abc class. 
    def __init__(self, file_path):
        self.file_path = file_path

try:
    invalid_loader = BrokenLoader("path/to/data.csv")
except TypeError as e:
    print(f"Successfully Caught the error for Borken Loader: {e}")