#pragma once

// Rendering macros

// Small offset added to ray origins.
#define BIAS 0.02f

// How many times primary rays can bounce before the path dies
#define BOUNCES 8

// How many samples to render per pass
#define SAMPLES 512

// How many passes to render
#define PASSES 8