
// Numerical_Calculation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <boost/math/quadrature/gauss_kronrod.hpp>
using namespace boost::math::quadrature;

double Precipitate::numericalIntegral()
{
	/*
    This function will calculate the numerical integral of a given function.
	It is a function from a Class with the name of "Precipitate".
	To define the function, internal variables of the class will be used.
	*/
	double integral1 = 0.;
	double diffCo = 0;
	diffCo = DSi;
	_a1AnswerSq = _a1Answer * _a1Answer;

	auto fun1 = [&](double x) {
		return exp(-x * x / (4. * diffCo * _t)) / (x * x - _e31Square * _a1AnswerSq);
	};

	integral1 = gauss_kronrod<double, 15>::integrate(fun1, _a1Answer, 200 * _a1Answer, 12, 1.0E - 6, &error1);
	return integral1;
}

/*
In the rest, three C++ functions are provided from the Precipitate class.
The first two functions, Precipitate::f1_Package() and Precipitate::f2_Package(),
calculate the value of Eq.1 and Eq.2 and their partial derivatives. 
Finally, the third function, Precipitate::Calculate(), will apply the Newton-Raphson method
*/

#include <boost/math/quadrature/gauss_kronrod.hpp>
using namespace boost::math::quadrature;
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