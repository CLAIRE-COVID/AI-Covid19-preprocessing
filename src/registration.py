import numpy as np
import pyelastix

#target and reference are assumed numpy arrays
#potential criticalities: different size. Here we assume they are at the same size.
def registration(to_register, reference):
	#internal elastix parameters
	params = pyelastix.get_default_params(type='AFFINE')
	params.NumberOfResolutions = 8
	params.AutomaticTransformInitialization = True
	params.AutomaticScalesEstimation = False
	params.NumberOfHistogramBins = 64
	params.MaximumStepLength = 5.0
	params.MaximumNumberOfIterations = 500

	registered, field = pyelastix.register(to_register, reference, params, verbose=0)
	return registered
