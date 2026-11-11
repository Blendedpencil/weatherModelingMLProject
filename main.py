
from transformers import AutoImageProcessor, SwinForPDE

processor = AutoImageProcessor.from_pretrained("camlab-ethz/Poseidon-L")
model = SwinForPDE.from_pretrained("camlab-ethz/Poseidon-L", device_map="auto")