#pragma once

// Rendering macros

// Small offset added to ray origins.
#define BIAS 0.02f

// How many times primary rays can bounce before the path dies
#define BOUNCES 10000

// How many samples to render per pass
#define SAMPLES 256

// How many passes to render.
#define PASSES 16

// Maximum chance of a path to survive
#define MAX_SURVIVAL 0.95f

// Minimum probability for generating a sample, to avoid nans
#define F_EPS 1.1920929e-7f