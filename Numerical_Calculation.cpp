/*
This C++ files is a part of large scale project on:
"Mathematical development and C++ implementation of a complex materials behavior for industrial applications"
The C++ implementation of this project is made of more than 6000 lines.
This object-orinetd coding is made of two main Classes: 
1) Precipitate Class
2) Equations Class
The first class will manage all the details for each precipitate phase that we have in our alloy.
The second class will control all the equations which should be applied on top of first class.

For confidentiality purposes, the full code is not shared at GitHub.
However, a definition of these two classes along with some of the numerical calculation functions including
numerical integral and the details of Newton-Raphson implementation are shaed here.
To implement the Newton-Raphson method, first development of mathematical formulation is done,
then numerical derivative of the required functions are measure by MATLAB. Finally, everything is embedded in 
the developed C++ model.
*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////                Adding required libraries to the project                 //////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <time.h>
#include <cmath>
#include <string>
#include <cmath>
#include <sstream> 
#include <math.h> 

//Including Boost Library, for the numereical integral:
#include <boost/math/quadrature/gauss_kronrod.hpp>
//Including mlpack Library, for mapping solubility and coefficient diffusion based on the phase diagram:
#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
using namespace boost::math::quadrature;
using namespace mlpack;
using namespace mlpack::tree;
using namespace arma;
using namespace mlpack::neighbor; // NeighborSearch and NearestNeighborSort
using namespace mlpack::metric; // ManhattanDistance
using namespace std;

#define _MATH_DEFINES_DEFINED
/* Definitions of useful mathematical constants
* M_E        - e
* M_PI       - pi
*/
#define M_E        2.71828
#define M_PI       3.14159
#define M_perfectGas 8.31446
//The unit of gas constant is J/mol.K


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////                Definition of Precipitate Class                 ///////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
class precipitate {
	//This class will manage all the details for each precipitate phase that we have in our alloy.
public:
	precipitate();//Constructor_1
	precipitate(double, double, double, double, double, double, double, double, double, double, double,
		double, double, double, double, double, double, double, double, double, double, double, double);//Constructor_2
	void setAnswer(double, double);
	void setvFrac(double, double);
	void setaVar(double aVar_);
	void setaVar2(double aVarDissolve_, double aVar2_);
	void setMob(double mob_);
	void setvC(double vC_);
	void setvC2(double vC2vCFun_);
	void setcSRef(double cSRef_);
	void setcSDepend(double cSdepend_);
	void setevolMode(int evolMode_);
	void setAppType(int appType_);
	int getAppType();
	void setStepCtr(int stepCtr_);
	int getStepCtr();
	void setIsoThermal(bool isoThermal_);
	bool getIsoThermal();
	void setRegFlag(bool regFlag_);
	bool getRegFlag();
	int getevolMode();
	double getaVar();
	double getaVar2();
	void setfDRatio(double ratio_);
	double getfDRatio();
	void setaCurrent(double aCurrent_);
	double getaCurrent();
	double getaRatio();
	double getNC();
	void settf(double t);
	double gettf();
	double getaCGrowth();
	double getaCDissolve();
	double getcoVC();
	double getcPer1(), getcPer2(), getcPer3();
	double getcEq1(), getcEq2(), getcEq3();
	int getsolCnt();
	void paramSetting(double temperature);
	void calc1(double&, double c0Var, double cPerVar);
	void calc2(double, double&);
	void mainSolution(int x1, bool& swKCalc, double soluteLst[16], double cBList1[5], double cBList2[5], double kC, double kC2, double diffusionList[5], double dt, double mob, double* coeffArray, double temVar, double timeVar, double hRateVar, double* res);
	void preSolution(int x1, int order, double soluteLst[16], double cBList[5], double kC, double diffusionList[5], double deltaTime, double tem, double mob, double* res);
	void findTimeRoutin1(int x1, int ctr, double tCurrentVar, double a1Var, double kCVar, double deltaTime, double& tAnswer, double& cStarAnswer, double& cSDependAnswer, bool& OptimumFound, double* coeffArray);
	void findTimeRoutin2(int x1, int order, double& f1, double& f2, double deltaTime, bool& optimumFound);
	int routFinding(double f1, double f2, double& answer1, double& answer2, double refValue, double& tResult, double& fResult);
	void regularization(double& tVar1, double& tVar2, double& f1, double& f2, double& coTime1, double& coTime2, double deltaTime);
	bool optimumCheck(int status, double f1value, double f2value, double t2value, double cSSol2, double cSDep2, double& fAddress, double& timeAddress, double& cSAddress, double& cDepAddress);
	double calcTime(int x1Var, double cStarVar);
	double calccSDepend(double cStarVar);
	void checkcStarLogic(int x1Var, double& cSRefGuess1, double& cSRefGuess2, double ceVar);
	void guessAnswer(int x1Var, double difVar, double cStarVar, double& cStarTemp1_1, double& cStarTemp1_2);
	void getc1(int x1Var, int i, double& c1);
	void gettimeVar(double timeVar, double* timeOut);
	void stopIfNan(double var1, string var2);
	void resetTime();
	string geometry;

private:
	int solCnt;
	double tf = -1;// For infinite system, we use -1 for indicator that tf is not defined already.
	double cPerSol1, cPerSol2, cPerSol3;		//molar fraction of Cu in precipitate.									
	double cEqSol1, cEqSol2, cEqSol3;
	double cSSol1, cSSol2, cSSol3;
	double cBSol1, cBSol2, cBSol3, cBSol1Next, cBSol2Next, cBSol3Next;
	double cBRefNow, cBRefNext, cBRef, cEqRef, dCoefRef, cPerRef;
	double cSRef;
	double cSDepend;
	double cBDepNow, cBDepNext, cBDepend, cEqDepend, dCoefDepend, cPerDepend;
	double prevcStar;
	double aCurrent;
	double aCGrowth, aCDissolve, a0;
	double coVC;
	double kC;
	double cStar;
	double vFrac, vFrac2;
	double aVar, aVar2, aVarDiss;
	double NC;						                //Heterogenous nucleation site(count / m3).
	double co1CInterface;
	double co2CInterface;
	double aRatioVar;
	double slope_aRatio;
	double intercept_aRatio;
	bool swDiss;
	double vFinal, v;
	double vC, vC2;
	double tVar, tVar1;
	double tRes1, tRes2;
	double mob, m0, m1, m2, m3, m4;  							    //unit: m4 / J.s
	double aRatio;
	double aRatio2;
	double e31Sq;
	double minSize1, minSize2;
	double deltaL;
	double vM;						                //unit: m3 / mol
	double intResValue;
	double coVolume;
	double slope_root;
	double intercept_root;
	double dt_root;
	bool swSolutionBasedcStar;
	double error1 = 0.;
	double error2 = 0.;
	double previous_Error1 = 10000.;
	double previous_Error2 = 10000.;
	double fDRatio = 0;
	double drivingForce = 0;
	double cSRefGuess1[4], cSRefGuess2[4], cSDepFound1[4], cSDepFound2[4];
	double time1Lst[4] = { 0 }, time2Lst[4] = { 0 };
	int evolMode = 1;//It is used for saving evolution mode: 1=growth or 2=dissolution.
	int appType = 1; //It is used for saving simulation type. 1=general, 2= zset, 3=welding.
	int stepCtr = 0;
	bool isothermal = false;
	bool regFlag = true;

	/*
	In the following section, we define some variables which only will be used as parameters in the calculation.
	Indeed, they are kind of general variables like temperature and diffusion.
	In addition, we will define only one variable for some parameters that each precipitates can have more than one
	like cEquilibrium in case of multicomponent precipitates. But it is correct, becaused we will use them only in
	one part of calculation, and they are not used to define precipitates generally.
	The reason behind defining such variables is that we consider some equations as functions of the precipitates CLASS.
	*/
	double x1, cPeri, cB, D, tem, tInterval;
	double dCoefSol1, dCoefSol2, dCoef3;
	double answer1, answer2;
	void cStarFunc();
	double evolutionFunc();
	void exitFunc(int mode, const char* var1, int var2);
	void slopeCalculator(double x1, double x2, double f1, double f2, double& deltat);
	void stopIfNegative(double var1, string var2);
};

precipitate::precipitate() {
	//Constructor_1 of the precipitate class:
	cSRef = 0;
	cSDepend = 0;
}

precipitate::precipitate(double m4_, double m3_, double m2_, double m1_, double m0_, double solCnt1_, double cEq1_, double cEq2_,
	double cEq3_, double cEq4_, double cEq5_, double solCnt2_, double cPer1_, double cPer2_, double cPer3_, double cPer4_, double cPer5_,
	double aRatio_, double NC_, double aCGrowth_, double aCDissolve_, double aCurrent_, double vM_) {
	//Constructor_2 of the precipitate class:
	cSRef = 0;
	cSDepend = 0;
	cStar = 0;
	kC = 0;
	answer1 = 0;
	answer2 = 0;
	m0 = m0_;
	m1 = m1_;
	m2 = m2_;
	m3 = m3_;
	m4 = m4_;
	solCnt = solCnt1_;
	cPerSol1 = cPer1_;
	cPerSol2 = cPer2_;
	cPerSol3 = cPer3_;
	cEqSol1 = cEq1_;
	cEqSol2 = cEq2_;
	cEqSol3 = cEq3_;
	aRatio = aRatio_;
	e31Sq = 1. - 1. / (aRatio_ * aRatio_);
	NC = NC_;
	aCGrowth = aCGrowth_;
	aCDissolve = aCDissolve_;
	aCurrent = aCurrent_;
	a0 = aCurrent_;
	vM = vM_;
	deltaL = 4.0e-11;
	swDiss = false;
	const double PI = 3.1416;
	coVolume = (NC * 4. * PI) / 3;
	aRatioVar = 1;
	swSolutionBasedcStar = true;

	/*
	Suppose we have:
	(x1, y1), (x2, y2)
	If: y=ax+b, THEN:
	a=(y2-y1)/(x2-x1);
	b=(x2*y1-x1*y2)/(x2-x1)
	*/
	slope_aRatio = (aRatio - aRatioVar) / (aCGrowth - aCurrent);
	intercept_aRatio = (aCGrowth * aRatioVar - aCurrent * aRatio) / (aCGrowth - aCurrent);
	/*
	We consider 2 variables minSize1 and minSize2 to controle the dissolution.
	It should be noted that we set aspect ratio equal to unity at the initial of simulation when the size is equal to minSize1.
	Therefore, during the dissolution as soon as we get minSize1 and beyond that we keep aspect ratio equal to one.
	In addition, minSize2 is the size which lower that existance of precipitate is physically meaningless.
	*/
	minSize1 = aCurrent;
	minSize2 = 0.2E-9;
}

void precipitate::regularization(double& tVar1, double& tVar2, double& f1, double& f2, double& coTime1, double& coTime2, double deltaTime) {
	/*
	Genral Routines:
	f1 is supposed to be negative,
	f2 is supposed to be positive.

	f1 as initial guess must be negative, but if f1>0:
	in growth:		we have to decrease time.
	in dissolution: we have to increase time.

	f2 as initial guess must be positive, but if f2<0:
	in growth:		we have to increase time.
	in dissolution: we have to decrease time.

	For the Regularization:
	I considered two steps regularization.
	Step 1: We modify tVar directly. It has more impact than second step of the regularization.
	Step 2: We modify coTime. It has less impacts than first step of the regularization.

	General Routine:
	Regularization of f1, if f1 is negative:
	in growth:		we increase time, maybe f1 still keeps its negative value. So we get closer to the root.
	in dissolution: we decrease time, maybe f1 still keeps its negative value. So we get closer to the root.

	Regularization of f2, if f2 is positive:
	in growth:		we decrease time, maybe f2 still keeps its positive value. So we get closer to the root.
	in dissolution: we increase time, maybe f2 still keeps its positive value. So we get closer to the root.
	*/

	bool justf1 = true;//By experiance I have found that it is better to apply regularization only on f1.
	double tVarSource1 = tVar1;
	double tVarSource2 = tVar2;
	double coTemp[2], fTemp[2], cSDepTemp[2], cSRefTemp[2], tVarTemp[2];
	int tempFlag = 0;
	if (f1 < 0 && f1 != -999) {
		/*
		As a regularization for f1, if f1 is negative :
		in growth :		 we increase time, maybe f1 still keeps its negative value.So we get closer to the root.
		in dissolution : we decrease time, maybe f1 still keeps its negative value.So we get closer to the root.
		*/

		//Step 1 of regularization: We modify tVar directly. It has more impact than second step of regularization.
		for (int k = 0; k < 2; k++) {
			if (evolMode == 1) {
				//We are at mixed-mode growth.
				tVarTemp[k] = (1. + (k + 1.) * 0.25) * tVar1;
			}
			else if (evolMode == 2) {
				//We are at mixed-mode dissolution.
				tVarTemp[k] = (1. - (k + 1.) * 0.25) * tVar1;
			}

			tVar = tVarTemp[k];
			if (tVar < 0)
				break;
			fTemp[k] = evolutionFunc();
			cSDepTemp[k] = cSDepend;
			cSRefTemp[k] = cSRef;
			if (k == 0) {
				if (fTemp[k] > 0 || fTemp[k] == -999) {
					tempFlag = 0;
					tVar1 = tVarSource1;
					tVar = tVarSource1;
					break;
				}
				else
					tempFlag = 1;
			}
			else if (k == 1) {
				if (fTemp[k] < 0 && fTemp[k] != -999)
					tempFlag = 2;
			}
		}
		if (tempFlag == 1 || tempFlag == 2) {
			int k = tempFlag - 1;
			f1 = fTemp[k];
			cSDepend = cSDepTemp[k];
			cSRef = cSRefTemp[k];
			tVar1 = tVarTemp[k];
			tVar = tVarTemp[k];
			tVarSource1 = tVarTemp[k];
		}

		//Step 2 of regularization: We modify coTime. It has less impacts than first step.
		if (coTime1 > 0.001) {
			tempFlag = 0;
			for (int k = 0; k < 2; k++) {
				if (evolMode == 1) {
					//We are at mixed-mode growth.
					coTemp[k] = (1. + (k + 1.) * 0.25) * coTime1;
				}
				else if (evolMode == 2) {
					//We are at mixed-mode dissolution.
					coTemp[k] = (1. - (k + 1.) * 0.25) * coTime1;
				}

				tVar1 = tVarSource1 - coTime1 * deltaTime;
				tVar1 += coTemp[k] * deltaTime;
				tVarTemp[k] = tVar1;
				tVar = tVar1;
				if (tVar < 0)
					break;
				fTemp[k] = evolutionFunc();
				cSDepTemp[k] = cSDepend;
				cSRefTemp[k] = cSRef;
				if (k == 0) {
					if (fTemp[k] > 0 || fTemp[k] == -999) {
						tempFlag = 0;
						break;
					}
					else
						tempFlag = 1;
				}
				else if (k == 1) {
					if (fTemp[k] < 0 && fTemp[k] != -999)
						tempFlag = 2;
				}
			}

			if (tempFlag == 1 || tempFlag == 2) {
				int k = tempFlag - 1;
				f1 = fTemp[k];
				cSDepend = cSDepTemp[k];
				cSRef = cSRefTemp[k];
				tVar1 = tVarTemp[k];
				tVar = tVarTemp[k];

				if (evolMode == 1) {
					//We are at mixed-mode growth.
					coTime1 = (1. + (k + 1.) * 0.25) * coTime1;
				}
				else if (evolMode == 2) {
					//We are at mixed-mode dissolution.
					coTime1 = (1. - (k + 1.) * 0.25) * coTime1;
				}
			}
		}
	}

	if (justf1 == false) {
		tempFlag = 0;
		if (f2 > 0) {
			/*
			As a regularization for f2, if f2 is negative :
			in growth :		 we increase time, maybe f2 still keeps its negative value.So we get closer to the root.
			in dissolution : we decrease time, maybe f2 still keeps its negative value.So we get closer to the root.
			*/
			//if (evolMode == 10) {
			//	//We are at mixed-mode growth.
			//	tVar2 -= coTime2 * deltaTime;
			//}
			//else if (evolMode <= 2) {
			//	//We are at mixed-mode dissolution.
			//	tVar2 += coTime2 * deltaTime;
			//}

			//Step 1 of regularization: We modify tVar directly. It has more impact than second step of regularization.
			for (int k = 0; k < 2; k++) {
				if (evolMode == 1) {
					//We are at mixed-mode growth.
					tVarTemp[k] = (1. - (k + 1.) * 0.25) * tVar2;
				}
				else if (evolMode == 2) {
					//We are at mixed-mode dissolution.
					tVarTemp[k] = (1. + (k + 1.) * 0.25) * tVar2;
				}

				tVar = tVarTemp[k];
				if (tVar < 0)
					break;
				fTemp[k] = evolutionFunc();
				cSDepTemp[k] = cSDepend;
				cSRefTemp[k] = cSRef;
				if (k == 0) {
					if (fTemp[k] < 0) {
						tempFlag = 0;
						tVar2 = tVarSource2;
						tVar = tVarSource2;
						break;
					}
					else
						tempFlag = 1;
				}
				else if (k == 1) {
					if (fTemp[k] > 0)
						tempFlag = 2;
				}
			}
			if (tempFlag == 1 || tempFlag == 2) {
				int k = tempFlag - 1;
				f2 = fTemp[k];
				cSDepend = cSDepTemp[k];
				cSRef = cSRefTemp[k];
				tVar2 = tVarTemp[k];
				tVar = tVarTemp[k];
				tVarSource2 = tVarTemp[k];
			}

			//Step 2 of regularization: We modify coTime. It has less impacts than first step.
			if (coTime2 > 0.001) {
				tempFlag = 0;
				for (int k = 0; k < 2; k++) {
					if (evolMode == 1) {
						//We are at mixed-mode growth.
						coTemp[k] = (1. - (k + 1.) * 0.25) * coTime2;
					}
					else if (evolMode == 2) {
						//We are at mixed-mode dissolution.
						coTemp[k] = (1. + (k + 1.) * 0.25) * coTime2;
					}

					tVar2 = tVarSource2 - coTime2 * deltaTime;
					tVar2 += coTemp[k] * deltaTime;
					tVarTemp[k] = tVar2;
					tVar = tVar2;
					if (tVar < 0)
						break;
					fTemp[k] = evolutionFunc();
					cSDepTemp[k] = cSDepend;
					cSRefTemp[k] = cSRef;
					if (k == 0) {
						if (fTemp[k] < 0) {
							tempFlag = 0;
							break;
						}
						else
							tempFlag = 1;
					}
					else if (k == 1) {
						if (fTemp[k] > 0)
							tempFlag = 2;
					}
				}

				if (tempFlag == 1 || tempFlag == 2) {
					int k = tempFlag - 1;
					f2 = fTemp[k];
					cSDepend = cSDepTemp[k];
					cSRef = cSRefTemp[k];
					tVar2 = tVarTemp[k];
					tVar = tVarTemp[k];

					if (evolMode == 1) {
						//We are at mixed-mode growth.
						coTime2 = (1. - (k + 1.) * 0.25) * coTime2;
					}
					else if (evolMode == 2) {
						//We are at mixed-mode dissolution.
						coTime2 = (1. + (k + 1.) * 0.25) * coTime2;
					}
				}
			}
		}
	}
}

bool precipitate::optimumCheck(int status, double f1value, double f2value, double t2value, double cSt2, double cSDep2, double& fAddress, double& timeAddress, double& cSAddress, double& cSDepAddress) {
	/*
	This function will evaluate the variables of f1Value and f2Value as the output of the calculations. 
	For that matter, there are some reference values that the outputs will be checked with them.
	Depending on input variable of "status" three different conditions will be checked:
	if status=1: Only f1Value will be checked.
	if status=2: Only f2Value will be checked.
	if status=3: Both f1Value and f2Value will be checked.
	The output is a boolean flag. 
	If the given condition is verified the output will be True, 
	otherwise it returns False.
	*/
	double refValue = 1.0E-18;

	if (status == 1) {
		//The condition will be checked only for f1 and time1.
		if (abs(f1value) < refValue)
			return true;
		return false;
	}

	else if (status == 2) {
		//The condition will be checked only for f2 and time2.
		if (abs(f2value) < refValue) {
			fAddress = f2value;
			timeAddress = t2value;
			cSAddress = cSt2;
			cSDepAddress = cSDep2;
			return true;
		}
		return false;
	}

	else if (status == 3) {
		//The condition will be checked for both (f1, time1) and (f2,time2).
		if (abs(f1value) < refValue) {
			if (abs(f2value) < refValue) {
				//Since both answers are verified, the minimum will be set in f1 and Time1.
				if (abs(f2value) < abs(f1value)) {
					fAddress = f2value;
					timeAddress = t2value;
					cSAddress = cSt2;
					cSDepAddress = cSDep2;
				}
			}
			return true;
		}
		else if (abs(f2value) < refValue) {
			fAddress = f2value;
			timeAddress = t2value;
			cSAddress = cSt2;
			cSDepAddress = cSDep2;
			return true;
		}
		return false;
	}
	cSSol1 = cSDepend;
	cSSol2 = cSRef;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////                Definition of Equations Class                 /////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
class equations {
	//This class will control all the equations which should be applied on top of first class.

public:
	/*
	p array:          list of all precipitates.
	We use maxdt as a limitation for maximum allowed dt at each time step.
	We set this value according to the heating rate.
	For example, at each time step we do not want the variation of temperature exeed 0.5 K. (maxdTem = 0.5)
	So, if the heating rate be 10 K/min -> dt shold not exceed 3 seconds.
	If the dt of precipitates are less than 3 seconds (maxdt = 3), we choose dt directly from the precipitates
	which has the minimum value of dt. But if dt of all precipitates are higher than 3 seconds,
	we set dt based on maxdt.
	mindt is only a variable that we use to find the minimum dt of precipitates.

	precLst1 saves calculated equilibrium values from phase diagram for all precipitates with 1 solutes.
	precLst2 saves calculated equilibrium values from phase diagram for all precipitates with 2 solutes.
	precLst3 saves calculated equilibrium values from phase diagram for all precipitates with 3 solutes.

	precCnt1 saves caounts of precipitates with 1 solutes.
	precCnt2 saves caounts of precipitates with 2 solutes.
	precCnt3 saves caounts of precipitates with 3 solutes.

	static variables are defined in the heap not stack, so we can have very large allocation for the required variables.
	varaibles without static keyword, are defined in the stack memory of the thread, so we have limitation for the maximum allocation.
	static variables are automatically filled by zero, so we don't need to make them equal to zero.

	swAllInterface is an indicator showing whether we are at the initial state of the model or not.
	Initial state of the model means the time that evolution of all precipitates are controlled by
	the kinetics of the interface.

	When the velocity of a precipitate, decrease alot, after a while, we should prevent calculation of kC.
	Otherwise calculation of kC leades to increasing of cStar which is not meaningful.
	For this reason, we define a switch array for K calculation and based on the this switch array,
	we decide to do or not to do the calculation for kC. We call this switch array swKLst.

	swEvolution: keep the state of evolution for all precipitates.
	sw = 0 -> interface growth,
	sw = 1 -> mixed-mode growth,
	sw = 2 -> mixed-mode dissolution,
	sw = 3 -> interface dissolution.
	t1: Start temperature, t2: Final temperature, temperature: Current temperature, hRate: Heating rate
	tLst: keeps temperature iteratives which covers heating/cooling range.
	equiCnt1: total count of matCalc calculattion for precipitates with 1 solute.
	equiCnt2: total count of matCalc calculattion for precipitates with 2 solutes.
	equiCnt3: total count of matCalc calculattion for precipitates with 3 solutes.
	*/

	equations();//Constructor
	void preparation1(double* data, double* inWord);
	void preparation2(double* c0Lst);
	void initialize(int applicationType, bool regFlag, double time1, double time2, double temperature1, double temperature2);
	void setGeometry();
	void calculate(int stepCTR, double time1, double time2, double temperature1, double temperature2);
	void exportData();
	void exportData2();
	void closeFiles();
	void prepareKNN();
	void searchMatCalc();
	double gettCurrent();
	bool getAgingBeforeWeld();
	int alloy = 6000;

private:
	ofstream outfile, outfile1, outfile2, knnfile, matScript, tempfile;
	int pptCnt, solCnt;
	double dTperSecond;
	double tem;
	double cbLst[5] = { 0 };
	double cbLst2[5] = { 0 };
	double diffLst[5] = { 0 };
	double diffLst2[5] = { 0 };
	int idMax;
	int precCnt1 = 0;
	int precCnt2 = 0;
	int precCnt3 = 0;
	double tem1, tem2, hRate, tIncrement;
	double t1, t2;
	double tCurrent;
	double tPrevious = 0;
	double temperature;
	double dt = 10;
	double dtSource;
	double mindt, maxdt;
	double resArray[5];
	bool swAllInterface = true;
	bool zsetFlag = false;
	bool weldingFlag = false;
	bool dscFlag = false;
	bool isoFlag = false;
	bool agingBeforeWeld = false;
	double requiredAgingTime = 0;

	static const int sectionCnt = 30;
	static const int maxPrec = 20;
	precipitate p[maxPrec];
	double coeffLst[8 * maxPrec];
	int swEvolution[maxPrec];
	bool swDissFirstStep[maxPrec];
	bool swDissExperiance[maxPrec];//We use it to check whether in the past the model calculates dissolution mode or not.
	double dissInterceptLst[maxPrec];
	bool swFirstStep[maxPrec];
	bool swSolutionBasedcStar;
	bool swKLst[maxPrec];
	double mobLst[maxPrec][6];
	double mobLst2[maxPrec];
	double precSolute[maxPrec][16];
	double precSolute2[maxPrec];
	double solutePrec[5][23];
	double vcCoefLst[maxPrec];
	double aCurrentLst[maxPrec];
	double aRatioLst[maxPrec];
	double vCLst[maxPrec];
	double vCLst2[maxPrec];
	double vCLst3[maxPrec];
	double kCLst[maxPrec] = { 0 };
	double kCLst2[maxPrec] = { 0 };
	double kCLst3[maxPrec] = { 0 };
	double vFinalLst[maxPrec] = { 0 };
	bool bypassLst[maxPrec] = { false };
	bool dissStartLst[maxPrec] = { 0 };
	double v1Lst[maxPrec] = { 0 };
	double da_dtLst[maxPrec] = { 0 };
	double aCLstGrowth[maxPrec];
	double aCLstDissolve[maxPrec];
	double aVarLst2[maxPrec];
	double aVarDissLst[maxPrec];
	double dlLst[maxPrec];
	double vFracLst[maxPrec];
	double vFracLst2[maxPrec];
	//The following array are used for knn model.
	// They will save the data which are optained from the phase diagram.
	double* lst1_p1_sol1 = new double[29160];
	double* lst1_p1_sol2 = new double[29160];
	double* lst1_p2_sol1 = new double[29160];
	double* lst1_p2_sol2 = new double[29160];
	double* lst1_p3_sol1 = new double[29160];
	double* lst1_p3_sol2 = new double[29160];
	double* lst1_Diff_sol1 = new double[29160];
	double* lst1_Diff_sol2 = new double[29160];

	double* lst2_p1_sol1 = new double[29160];
	double* lst2_p1_sol2 = new double[29160];
	double* lst2_p2_sol1 = new double[29160];
	double* lst2_p2_sol2 = new double[29160];
	double* lst2_p3_sol1 = new double[29160];
	double* lst2_p3_sol2 = new double[29160];
	double* lst2_Diff_sol1 = new double[29160];
	double* lst2_Diff_sol2 = new double[29160];

	double* lst3_p1_sol1 = new double[29160];
	double* lst3_p1_sol2 = new double[29160];
	double* lst3_p2_sol1 = new double[29160];
	double* lst3_p2_sol2 = new double[29160];
	double* lst3_p3_sol1 = new double[29160];
	double* lst3_p3_sol2 = new double[29160];
	double* lst3_Diff_sol1 = new double[29160];
	double* lst3_Diff_sol2 = new double[29160];

	double* lst4_p1_sol1 = new double[29160];
	double* lst4_p1_sol2 = new double[29160];
	double* lst4_p2_sol1 = new double[29160];
	double* lst4_p2_sol2 = new double[29160];
	double* lst4_p3_sol1 = new double[29160];
	double* lst4_p3_sol2 = new double[29160];
	double* lst4_Diff_sol1 = new double[29160];
	double* lst4_Diff_sol2 = new double[29160];

	double* lst5_p1_sol1 = new double[29160];
	double* lst5_p1_sol2 = new double[29160];
	double* lst5_p2_sol1 = new double[29160];
	double* lst5_p2_sol2 = new double[29160];
	double* lst5_p3_sol1 = new double[29160];
	double* lst5_p3_sol2 = new double[29160];
	double* lst5_Diff_sol1 = new double[29160];
	double* lst5_Diff_sol2 = new double[29160];
	//cSList row: Precipitate
	//cSList Columns: Solute 
	double cSLst[maxPrec][3] = { 0 };
	double dtLst[maxPrec];
	double cBarLst[30] = { 0 };//It is only used for preparation. After thet, cbLst will be used. 
	int solute1Cnt = 0;
	int solute2Cnt = 0;
	int solute3Cnt = 0;
	int solute4Cnt = 0;
	int solute5Cnt = 0;
	double solute1Lst[41];
	double solute2Lst[41];
	double solute3Lst[41];
	double solute4Lst[41];
	double solute5Lst[41];
	int binaryCnt = 0;
	int ternaryCnt = 0;
	int fournaryCnt = 0;
	int binaryLst[20];
	int ternaryLst[20];
	int fournaryLst[20];
	typedef NeighborSearch<NearestNeighborSort, EuclideanDistance, mat, KDTree> KNN;
	KNN knnModel;

	float round(double);
	void growthFlag(int code, int* flagArr);
	void vFracFunc();
	void vCFunc(int sw, double hRateVar);
	void kCFunc(int sw);
	int cBFunc(int sw);
	void diffFunc(int sw, double hRateVar, int, double*, double var[][4]);
	void mobFunc(int sw);
	void exitFunc(int mode, const char* var1, int var2);
	void setTempValues(int flag, int rowId, double& temp1, double& temp2, double& temp3, double& temp4, double& temp5, double& temp6, double& temp7, double& temp8);
	void finishingPhase(int i);

	double deltacBarFunc(int pNumber, double x, double temperature);
	void signChange(int pNumber, double x, double temperature, double& x1, double& x2, double& f1, double& f2);
	double rootFind(int pNumber, double guess, double temperature, double& xResult, double& fResult);
	void solubilityFunc(int pNumber, double deltaCVar, double& solubility1, double& solubility2);
	void solubilityExtrapolated(int pNumber, double temperature, double& solubility1, double& solubility2);
};

equations::equations() {
	//Construtcor function
	ifstream matfile("../data/matcalc.txt");
	outfile1.open("../data/enthalpy_Matrix.mcs");
	outfile2.open("../data/enthalpy_Precipitate.mcs");
	knnfile.open("../data/knnfile.txt");
	matScript.open("../data/matScript.mcs");

	string line1_;
	while (!matfile.eof()) {
		getline(matfile, line1_);

		if (line1_.find("SET_FUNCTION_EXPRESSION") != string::npos) {
			if (line1_.find("X$FCC_A1") != string::npos || line1_.find("D$FCC_A1") != string::npos) {

				// These txt lines are usefule for calculation of solubility and diffusion; We ignore them in calculation of enthalpy.
				matScript << line1_ << "\n";
			}
			else {
				outfile1 << line1_ << "\n";
				outfile2 << line1_ << "\n";
				matScript << line1_ << "\n";
			}
		}
		else {
			outfile1 << line1_ << "\n";
			outfile2 << line1_ << "\n";
			matScript << line1_ << "\n";
		}
	}

	outfile.open("../data/result.txt");

	idMax = 0;
	for (int i = 0; i < maxPrec; i++) {
		dlLst[i] = 2.0E-11;
		dtLst[i] = 100;
		swEvolution[i] = 0;
		swDissFirstStep[i] = false;
		swDissExperiance[i] = false;
		dissInterceptLst[i] = 0;
		// We are at growth mode.
		p[i].setevolMode(1);
		for (int j = 0; j < 8; j++) {
			coeffLst[8 * i + j] = 1;
		}
	}

	for (int i = 0; i < 41; i++) {
		solute1Lst[i] = -1;
		solute2Lst[i] = -1;
		solute3Lst[i] = -1;
		solute4Lst[i] = -1;
		solute5Lst[i] = -1;
	}

	for (int i = 0; i < 20; i++) {
		binaryLst[i] = -1;
		ternaryLst[i] = -1;
		fournaryLst[i] = -1;
	}
	swSolutionBasedcStar = true;
}

int equations::cBFunc(int sw) {
	/*
	This function will calculate the concentration of solutes in the matrix or cBar.
	As an input it only accepts sw as a switch:
	When sw = 1, it calculates the cBar	for Current Size of Particles.
	When sw != 1, it calculates the cBar for the next Size of Particles.

	As an output it returns one integer which describes the status of calculation.
	When return= 0, the calculation is not successfull. An error happens due to the fact that cBar gets negative.
	While as a boundary condition, cBar is supposed to be positive.
	When return= 1, the calculation is done successfully. The results are saved in internal arrays of "obj" as an instance of equations class.
	*/
	double c0Var = 0;
	int pCnt = 0;
	double sumvFrac = 0;
	int p_id = 0;
	double cPer = 0;
	double sigma_vF_cPer = 0;
	double vF = 0;

	if (sw == 1) {
		//For Current Size of Particles 
		for (int i = 0; i < solCnt; i++) {
			c0Var = solutePrec[i][0];
			pCnt = solutePrec[i][1];
			for (int j = 0; j < pCnt; j++) {
				p_id = solutePrec[i][2 + j];
				cPer = solutePrec[i][pCnt + 2 + j];
				vF = vFracLst[p_id];
				sigma_vF_cPer += cPer * vF;
				sumvFrac += vF;
			}
			cbLst[i] = (c0Var - sigma_vF_cPer) / (1 - sumvFrac);
			sigma_vF_cPer = 0;
			sumvFrac = 0;
			if (cbLst[i] < 0) {
				exitFunc(2, "cBar value", 1);
				return 0;
			}
		}
	}
	else {
		//For Next Size of Particles
		for (int i = 0; i < solCnt; i++) {
			c0Var = solutePrec[i][0];
			pCnt = solutePrec[i][1];
			for (int j = 0; j < pCnt; j++) {
				p_id = solutePrec[i][2 + j];
				cPer = solutePrec[i][pCnt + 2 + j];
				vF = vFracLst2[p_id];
				sigma_vF_cPer += cPer * vF;
				sumvFrac += vF;
			}
			cbLst2[i] = (c0Var - sigma_vF_cPer) / (1 - sumvFrac);
			sigma_vF_cPer = 0;
			sumvFrac = 0;
			if (cbLst2[i] < 0) {
				return 0;
			}
		}
	}
	return 1;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////                Numerical Calculation Functions                 ///////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
double numericalIntegral(double diffCo, double _a1Answer, double _t, double _e31Square)
{
	/*
    This function will calculate the numerical integral of a given function.
	*/
	double integral1 = 0.;
	double error1;
	double _a1AnswerSq = _a1Answer * _a1Answer;

	auto fun1 = [&](double x) {
		return exp(-x * x / (4. * diffCo * _t)) / (x * x - _e31Square * _a1AnswerSq);
	};

	integral1 = gauss_kronrod<double, 15>::integrate(fun1, _a1Answer, 200 * _a1Answer, 12, 1.0E-6, &error1);
	return integral1;
}

/*
In the rest, three C++ functions are provided from the Precipitate class.
The first two functions, Precipitate::f1_Package() and Precipitate::f2_Package(),
calculate the value of Eq.1 and Eq.2 and their partial derivatives. 
Finally, the third function, Precipitate::Calculate(), will apply the Newton-Raphson method
*/
void Precipitate::f1_Package()
{
	/*
	It is a function from a Class with the name of "Precipitate".
	To define the function, internal variables of the class will be used.
	This function will calculate the value of Eq.1 and its partial derivatives.
	*/
	_a1AnswerSq = _a1Answer * _a1Answer;
	_a1AnswerQb = _a1AnswerSq * _a1Answer;
	double DMgSq = DMg * DMg;

	double coConcentration = 0.;
	double rightHandSide = 0.;
	double diffCo = 0;

	double fDSi = 0.;
	double fDMg = 0.;
	double fDSitoMg = 0.;

	double integralRes1Mg = 0.;
	double integralRes2Mg = 0.;
	double integralRes1Si = 0.;
	double integralRes2Si = 0.;

	/*
	fun1 and fun3 are defined as lambda functions, and the variables are called by the reference.
	These two statements will be used as the internal term of the numerical integral calculation.
	*/
	auto fun1 = [&](double x) {
		//In the denominator of the following function, (-1) has bee multiplied.
		//Because in all of the integral terms within the derivative function, there is a (-1).
		//As a result, we use it directly in the derivative functions.
		//But we have to consider (-1) in the calculation of f(D).
		//Since the f(D) has: (x * x - _e31Square * _a1AnswerSq)
		return exp(-x * x / (4. * diffCo * _t)) / (_e31Square * _a1AnswerSq - x * x);
	};

	auto fun3 = [&](double x) {
		return -(2. * _a1Answer * _e31Square) * exp(-x * x / (4. * diffCo * _t)) / pow((_e31Square * _a1AnswerSq - x * x), 2.);
	};

	diffCo = DSi;
	integralRes1Si = gauss_kronrod<double, 15>::integrate(fun1, _a1Answer, 200 * _a1Answer, 12,	1.0E-6, &error1);

	diffCo = DSi;
	integralRes2Si = gauss_kronrod<double, 15>::integrate(fun2, _a1Answer, 200 * _a1Answer, 12,	1.0E-6, &error1);

	diffCo = DMg;
	integralRes1Mg = gauss_kronrod<double, 15>::integrate(fun1, _a1Answer, 200 * _a1Answer, 12,	1.0E-6, &error1);

	diffCo = DMg;
	integralRes2Mg = gauss_kronrod<double, 15>::integrate(fun2, _a1Answer, 200 * _a1Answer, 12,	1.0E-6, &error1);

	fDSitoMg = ((DSi * exp(-_a1AnswerSq / (DSi_4 * _t))) / (-integralRes1Si)) / 
		((DMg * exp(-_a1AnswerSq / (DMg_4 * _t))) / (-integralRes1Mg));

	setfDRatio(fDSitoMg);

	double cSSi = exp(_LNcStarSi_Answer);
	coConcentration = (cSSi - _cBarrSi1) / (_cPeriSi - cSSi);

	cSMg = (_cBarrMg1 + _cPeriMg * fDSitoMg * coConcentration) / (1. + fDSitoMg * coConcentration);

	rightHandSide = ((_a1Answer * _vMParticle) / (2. * _kC * pow(_t, 1.5)
		* _mobility * _perfectGas * _Temperature))
		+ (_cPeriMg * log(_cEquiMg) + _cPeriSi * log(_cEquiSi));

	_f1 = _cPeriMg * log(cSMg) + _cPeriSi * _LNcStarSi_Answer - rightHandSide;

	//Derivative of function1 based on a1
	//It is obtained using MATLAB. 
	df1_x1 = -_vMParticle / (2 * _Temperature * _kC * _mobility * _perfectGas * pow(_t, 1.5)) - (_cPeriMg * exp((_a1AnswerSq * (DMg - DSi)) / (4 * DMg * DSi * _t)) * exp(-_a1AnswerSq / (4 * DMg * _t)) * exp(-_a1AnswerSq / (4 * DSi * _t)) * (_cBarrSi1 - cSSi) * (_cPeriSi - cSSi) * (_cBarrMg1 - _cPeriMg) * (2 * DMg * DSi * _t * exp(_a1AnswerSq / (4 * DMg * _t)) * integralRes1Mg - 2 * DMg * DSi * _t * exp(_a1AnswerSq / (4 * DSi * _t)) * integralRes1Si + DMg * _a1AnswerQb * exp(_a1AnswerSq / (4 * DMg * _t)) * exp(_a1AnswerSq / (4 * DSi * _t)) * integralRes1Mg * integralRes1Si - DSi * _a1AnswerQb * exp(_a1AnswerSq / (4 * DMg * _t)) * exp(_a1AnswerSq / (4 * DSi * _t)) * integralRes1Mg * integralRes1Si - DMg * _a1AnswerQb * _e31Square * exp(_a1AnswerSq / (4 * DMg * _t)) * exp(_a1AnswerSq / (4 * DSi * _t)) * integralRes1Mg * integralRes1Si + DSi * _a1AnswerQb * _e31Square * exp(_a1AnswerSq / (4 * DMg * _t)) * exp(_a1AnswerSq / (4 * DSi * _t)) * integralRes1Mg * integralRes1Si - 2 * DMg * DSi * _a1AnswerSq * _t * exp(_a1AnswerSq / (4 * DMg * _t)) * exp(_a1AnswerSq / (4 * DSi * _t)) * integralRes3Mg * integralRes1Si + 2 * DMg * DSi * _a1AnswerSq * _t * exp(_a1AnswerSq / (4 * DMg * _t)) * exp(_a1AnswerSq / (4 * DSi * _t)) * integralRes3Si * integralRes1Mg + 2 * DMg * DSi * _a1AnswerSq * _e31Square * _t * exp(_a1AnswerSq / (4 * DMg * _t)) * exp(_a1AnswerSq / (4 * DSi * _t)) * integralRes3Mg * integralRes1Si - 2 * DMg * DSi * _a1AnswerSq * _e31Square * _t * exp(_a1AnswerSq / (4 * DMg * _t)) * exp(_a1AnswerSq / (4 * DSi * _t)) * integralRes3Si * integralRes1Mg)) / (2 * _t * (_a1AnswerSq * _e31Square - _a1AnswerSq) * (DSi * _cBarrSi1 * _cPeriMg * integralRes1Mg - DSi * _cPeriMg * cSSi * integralRes1Mg - DMg * _cBarrMg1 * _cPeriSi * exp((_a1AnswerSq * (DMg - DSi)) / (4 * DMg * DSi * _t)) * integralRes1Si + DMg * _cBarrMg1 * exp((_a1AnswerSq * (DMg - DSi)) / (4 * DMg * DSi * _t)) * cSSi * integralRes1Si) * (DSi * cSSi * integralRes1Mg - DSi * _cBarrSi1 * integralRes1Mg - DMg * exp((_a1AnswerSq * (DMg - DSi)) / (4 * DMg * DSi * _t)) * cSSi * integralRes1Si + DMg * _cPeriSi * exp((_a1AnswerSq * (DMg - DSi)) / (4 * DMg * DSi * _t)) * integralRes1Si));

	//Derivative of function1 based on LNcStarSi
	//It is obtained using MATLAB.
	df1_x2 = _cPeriSi - (DMg * DSi * _cPeriMg * cSSi * integralRes1Mg * integralRes1Si * (_cBarrSi1 - _cPeriSi) * (_cBarrMg1 * exp((_a1AnswerSq * (DMg - DSi)) / (4 * DMg * DSi * _t)) - _cPeriMg * exp((_a1AnswerSq * (DMg - DSi)) / (4 * DMg * DSi * _t)))) / ((DSi * _cBarrSi1 * integralRes1Mg + DMg * exp(((DMg - DSi) * _a1AnswerSq) / (4 * DMg * DSi * _t) + _LNcStarSi_Answer) * integralRes1Si - DSi * cSSi * integralRes1Mg - DMg * _cPeriSi * exp((_a1AnswerSq * (DMg - DSi)) / (4 * DMg * DSi * _t)) * integralRes1Si) * (DSi * _cPeriMg * cSSi * integralRes1Mg - DSi * _cBarrSi1 * _cPeriMg * integralRes1Mg - DMg * _cBarrMg1 * exp(((DMg - DSi) * _a1AnswerSq) / (4 * DMg * DSi * _t) + _LNcStarSi_Answer) * integralRes1Si + DMg * _cBarrMg1 * _cPeriSi * exp((_a1AnswerSq * (DMg - DSi)) / (4 * DMg * DSi * _t)) * integralRes1Si));
}


void Precipitate::f2_Package()
{
	/*
	It is a function from a Class with the name of "Precipitate".
	To define the function, internal variables of the class will be used.
	This function will calculate the value of Eq.2 and its partial derivatives.
	*/
	_a1AnswerSq = _a1Answer * _a1Answer;
	_a1AnswerQb = _a1AnswerSq * _a1Answer;

	double integralRes1Si = 0.;
	double integralRes2Si = 0.;
	double integralRes3Si = 0.;

	double diffCo = 0;
	using namespace boost::math::quadrature;
	auto fun1 = [&](double x) {
		return exp(-x * x / (4. * diffCo * _t)) / (x * x - _e31Square * _a1AnswerSq);
	};

	auto fun2 = [&](double x) {
		return (2. * _a1Answer * _e31Square) * exp(-x * x / (4. * diffCo * _t)) / pow((_e31Square
			* _a1AnswerSq - x * x), 2.);
	};

	diffCo = DSi;
	integralRes1Si = gauss_kronrod<double, 15>::integrate(fun1, _a1Answer, 200 * _a1Answer, 12,	1.0E-6, &error1);

	diffCo = DSi;
	integralRes2Si = gauss_kronrod<double, 15>::integrate(fun2, _a1Answer, 200 * _a1Answer, 12,	1.0E-6, &error1);

	double cSSi = exp(_LNcStarSi_Answer);
	_f2 = (_cBarrSi1 - cSSi) / (_cPeriSi - cSSi) - (_a1AnswerQb * exp(_a1AnswerSq /
	(4. * DSi * _t)) * (integralRes1Si)) / (2. * DSi * pow(_aspectRatio, 2.) * _t);

	//Derivative of function2 based on a1
	//It is obtained using MATLAB.
	df2_x1 = -(3. * _a1AnswerSq * exp(_a1AnswerSq / (4. * DSi * _t)) * integralRes1Si) / (2. * DSi * pow(_aspectRatio, 2.) * _t) - (_a1AnswerSq * _a1AnswerSq * exp(_a1AnswerSq / (4. * DSi * _t)) * integralRes1Si) / pow((2. * DSi * _aspectRatio * _t), 2.) - (_a1AnswerQb * exp(_a1AnswerSq / (4. * DSi * _t)) * (integralRes3Si + exp(-_a1AnswerSq / (4. * DSi * _t)) / (_a1AnswerSq * _e31Square - _a1AnswerSq))) / (2. * DSi * pow(_aspectRatio, 2.) * _t);

	//Derivative of function2 based on LNcStarSi
	//It is obtained using MATLAB.
	df2_x2 = cSSi * (_cBarrSi1 - _cPeriSi) / pow((_cPeriSi - cSSi), 2.);
}

double Precipitate::Calculate()
{
	/*
	It is a function from a Class with the name of "Precipitate".
	To define the function, internal variables of the class will be used.
	This function will apply Newton-Raphson method.
	*/
	bool sw = true;
	while (sw == true)
	{
		f1_Package();
		f2_Package();
		double newa1 = 0.;
		double newcStar = 0.;
		double det = 1. / (df1_x1 * df2_x2 - df1_x2 * df2_x1);
		inversejacobian[0] = det * df2_x2;
		inversejacobian[1] = -det * df1_x2;
		inversejacobian[2] = -det * df2_x1;
		inversejacobian[3] = det * df1_x1;

		newa1 = _a1Answer - (inversejacobian[0] * _f1 + inversejacobian[1] * _f2);
		newcStar = _LNcStarSi_Answer - (inversejacobian[2] * _f1 + inversejacobian[3] * _f2);

		error1 = abs((newa1 - _a1Answer) / newa1) * 100.;
		error2 = abs((newcStar - _LNcStarSi_Answer) / newcStar) * 100.;

		_a1Answer = newa1;
		_LNcStarSi_Answer = newcStar;

		if (error1 < 2.0e-08 && error2 < 2.0e-08)
		{
			sw = false;
			_a1Result = newa1;
			_LNcStarSi_Result = newcStar;
			_cStarSi_Result = exp(_LNcStarSi_Result);
			_cStarMg_Result = (_cBarrMg1 + _cPeriMg * _funcDSitoMg * (_cStarSi_Result
       				          - _cBarrSi1) / (_cPeriSi - _cStarSi_Result))
				              / (1. + _funcDSitoMg * ((_cStarSi_Result - _cBarrSi1)
					          / (_cPeriSi - _cStarSi_Result)));
			_error1Result = error1;
			_error2Result = error2;
		}
		previous_Error1 = error1;
		previous_Error2 = error2;
	}
	return 0;
}