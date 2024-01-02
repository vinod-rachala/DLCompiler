import numpy as np
from PyCompileAndRuntime import OMCompileExecutionSession

# Load onnx model and create OMCompileExecutionSession object.
inputFileName = './mnist.onnx'
# Set the full name of compiled model
sharedLibPath = './mnist.so'
# Set the compile option as "-O3"
session = OMCompileExecutionSession(inputFileName,sharedLibPath,"-O3")

# Print the models input/output signature, for display.
# Signature functions for info only, commented out if they cause problems.
session.print_input_signature()
session.print_output_signature()

# Do inference using the default entry point.
a = np.full((1, 1, 28, 28), 1, np.dtype(np.float32))
outputs = session.run(input=[a])

for output in outputs:
    print(output.shape)
