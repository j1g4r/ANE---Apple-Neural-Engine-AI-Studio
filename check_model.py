import coremltools as ct
import numpy as np
import sys
import os

model_path = sys.argv[1]
try:
    model = ct.models.MLModel(model_path)
    print(f"\nModel: {model_path}")
    print("\nInputs:")
    for input_desc in model.input_description:
        print(f"  {input_desc}: {model.get_spec().description.input[0].type.multiArrayType.shape}")
    
    print("\nOutputs:")
    for output_desc in model.output_description:
        # Try to find the output shape in the spec
        for out in model.get_spec().description.output:
            if out.name == output_desc:
                print(f"  {output_desc}: {out.type.multiArrayType.shape}")

except Exception as e:
    print(f"Error: {e}")
