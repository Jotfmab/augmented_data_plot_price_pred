from zenml.client import Client

# Get the active ZenML client
client = Client()

# Get the experiment tracker component from the active stack
experiment_tracker = client.active_stack.experiment_tracker

# Print its URI. This is the path MLflow is writing to.
print(experiment_tracker.get_tracking_uri())