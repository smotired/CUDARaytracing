#pragma once

// Rendering macros

// Small offset added to ray origins.
#define BIAS 0.0002f

// How many times primary rays can bounce or split
#define BOUNCES 32

// Minimum sample count for adaptive sampling
#define SAMPLE_MIN 4

// Maximum sample count for adaptive sampling
#define SAMPLE_MAX 64