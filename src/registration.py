import numpy as np
import pyelastix

#target and reference are assumed numpy arrays
#potential criticalities: different size. Here we assume they are at the same size.
def registration(to_register, reference):
	#internal elastix parameters
	params = pyelastix.get_default_params(type='RIGID')
	params.NumberOfResolutions = 8
	params.AutomaticTransformInitialization = True
	params.AutomaticScalesEstimation = False
	params.NumberOfHistogramBins = 64
	params.MaximumStepLength = 5.0
	params.MaximumNumberOfIterations = 2000

	registered, field = pyelastix.register(to_register, reference, params)
	return registered
