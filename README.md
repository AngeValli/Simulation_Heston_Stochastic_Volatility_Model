# Simulation_Heston_Stochastic_Volatility_Model
Simulation of the Heston Stochastic Volatility Model based on :
* Van der Stoep, A. W., Grzelak, L. A., & Oosterlee, C. W. (2014). The Heston stochastic-local volatility model: Efficient Monte Carlo simulation. International Journal of Theoretical and Applied Finance, 17(07), 1450045.
* Andersen, L. B. (2007). Efficient simulation of the Heston stochastic volatility model. Available at SSRN 946405.

## Description of the files & folder
	* class_Matrix_header.h & class_Matrix.cpp: files containing the definition of matrix class and usual operations. Contains methods to perform cholesky decomposition, (reduced) row echelon matrix transformation, thomas system

	* folder data: folder containing raw data to compute surface volatility
    	* option_implied_vol.csv
    	* option_prices.csv
    	* tenor_vol_chain.csv
    	* test_vol_curve.csv

	* class_tenor_cubicspline_header.h & class_tenor_cubicspline.cpp: files containing the build of tenor cubicspline method
	* class_surface_vol_header.h & class_surface_vol.cpp: files containing the build of surface volatility method
	* main_read_from_data.cpp_temp: main file to read raw data files and build surface volatility

	* class_Random_header.h & class_Random.cpp: files containing the definition of uniform and gaussian random variables
	* class_Model2F_header.h & class_Model2F.cpp: files containing the general definition of a 2 Factor model
	* class_PathSimulatorEuler2F_header.h & class_PathSimulatorEuler2F.cpp: files containing the definition of path simulator via Euler method for a 2 Factor model
	* class_HestonModel_header.h & class_HestonModel.cpp: files containing the definition of Heston model
	* main_heston.cpp_temp: main file to simulate the Heston model

## To execute
Rename the file *main_read_from_data.cpp_temp* or *main_heston.cpp_temp* into *main.cpp* if you respectively want to compute the volatility surface based on raw data or if you want to simulate the path from Heston Model.
Then, on Ubuntu, execute `g++ *.cpp -o main` to create executable file `main` and run it with `./main` in terminal.
