# fit_solarcell

The purpose of this script is to fit simple 1- to 3-diode models to given solar cell IV characteristic, possibly augmented with dark characteristics, too. It will only work when the model can reasonably reproduce the provided data. Note that the script is under development and might need tweaking.

Known issues
============

The input data must be in the 4th quadrant for light IV, and in the 1st for dark IV.
Current must be provided in mA.
