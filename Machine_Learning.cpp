void equations::prepareKNN() {

	// Vector of string to save tokens 
	vector <string> tokens1, tokens2, tokens3, tokens4, tokens5;
	string curWord;
	stringstream check1;

	mat Xtrain, ytrain1, ytrain2, ytrain3, ytrain4, ytrain5, ytrain_diff;
	bool loaded1 = data::Load("../data/knn_input.csv", Xtrain);
	bool loaded2 = data::Load("../data/phase_diagram1.csv", ytrain1);
	bool loaded3 = data::Load("../data/phase_diagram2.csv", ytrain2);
	bool loaded4 = data::Load("../data/phase_diagram3.csv", ytrain3);
	bool loaded5 = data::Load("../data/phase_diagram4.csv", ytrain4);
	bool loaded6 = data::Load("../data/phase_diagram5.csv", ytrain5);


	if (!loaded1 || !loaded2 || !loaded3 || !loaded4 || !loaded5 || !loaded6) {
		cout << "Error in reading input files!";
	}

	/*
	Pay Attention:
	in mat data type row and column will be transpose.
	due to simplicity of linear algebra calculation.
	SO in ytrain(i, j) --> i: column
							j: row


	In the train dataset, the last two columns are the responses variables:
		solubility of solute_1
		solubility of solute_2
	We split these two columns in two different varaibles, y_train_sol1, y_train_sol2
	IN case of Beta family precipitates, mg is considered sol1 and Si as sol2.

	We define two matrices (nei_Index and nei_Distance) which we will keep information of the neighbor points.
	nei_Index keeps the index of of neighbor points of the query point.
	nei_Distance keeps the distances of neighbor points from the query point.

	Since, it is a regression model, we will calculate the response by making uniform averaging.
	The nearest point to each point from test data set are saved in neighborindex array.
	In the following code: k represents the k nearest neighbor.
	if uniformAve==1 then we do just normal averaging: res=(y1+y2)/2
	if uniformAve==0 then we do averaging based on the distance.
	*/

	Row<double> ytrain1_p1_sol1, ytrain1_p1_sol2, ytrain1_p2_sol1, ytrain1_p2_sol2, ytrain1_p3_sol1, ytrain1_p3_sol2;
	Row<double> ytrain1_diff_sol1, ytrain1_diff_sol2;//Variables for diffusion coefficient
	ytrain1_diff_sol2 = conv_to<Row<double>>::from(ytrain1.row(ytrain1.n_rows - 1));
	ytrain1_diff_sol1 = conv_to<Row<double>>::from(ytrain1.row(ytrain1.n_rows - 2));
	ytrain1_p3_sol2 = conv_to<Row<double>>::from(ytrain1.row(ytrain1.n_rows - 3));
	ytrain1_p3_sol1 = conv_to<Row<double>>::from(ytrain1.row(ytrain1.n_rows - 4));
	ytrain1_p2_sol2 = conv_to<Row<double>>::from(ytrain1.row(ytrain1.n_rows - 5));
	ytrain1_p2_sol1 = conv_to<Row<double>>::from(ytrain1.row(ytrain1.n_rows - 6));
	ytrain1_p1_sol2 = conv_to<Row<double>>::from(ytrain1.row(ytrain1.n_rows - 7));
	ytrain1_p1_sol1 = conv_to<Row<double>>::from(ytrain1.row(ytrain1.n_rows - 8));


	Row<double> ytrain2_p1_sol1, ytrain2_p1_sol2, ytrain2_p2_sol1, ytrain2_p2_sol2, ytrain2_p3_sol1, ytrain2_p3_sol2;
	Row<double> ytrain2_diff_sol1, ytrain2_diff_sol2;//Variables for diffusion coefficient
	ytrain2_diff_sol2 = conv_to<Row<double>>::from(ytrain2.row(ytrain2.n_rows - 1));
	ytrain2_diff_sol1 = conv_to<Row<double>>::from(ytrain2.row(ytrain2.n_rows - 2));
	ytrain2_p3_sol2 = conv_to<Row<double>>::from(ytrain2.row(ytrain2.n_rows - 3));
	ytrain2_p3_sol1 = conv_to<Row<double>>::from(ytrain2.row(ytrain2.n_rows - 4));
	ytrain2_p2_sol2 = conv_to<Row<double>>::from(ytrain2.row(ytrain2.n_rows - 5));
	ytrain2_p2_sol1 = conv_to<Row<double>>::from(ytrain2.row(ytrain2.n_rows - 6));
	ytrain2_p1_sol2 = conv_to<Row<double>>::from(ytrain2.row(ytrain2.n_rows - 7));
	ytrain2_p1_sol1 = conv_to<Row<double>>::from(ytrain2.row(ytrain2.n_rows - 8));


	Row<double> ytrain3_p1_sol1, ytrain3_p1_sol2, ytrain3_p2_sol1, ytrain3_p2_sol2, ytrain3_p3_sol1, ytrain3_p3_sol2;
	Row<double> ytrain3_diff_sol1, ytrain3_diff_sol2;//Variables for diffusion coefficient
	ytrain3_diff_sol2 = conv_to<Row<double>>::from(ytrain3.row(ytrain3.n_rows - 1));
	ytrain3_diff_sol1 = conv_to<Row<double>>::from(ytrain3.row(ytrain3.n_rows - 2));
	ytrain3_p3_sol2 = conv_to<Row<double>>::from(ytrain3.row(ytrain3.n_rows - 3));
	ytrain3_p3_sol1 = conv_to<Row<double>>::from(ytrain3.row(ytrain3.n_rows - 4));
	ytrain3_p2_sol2 = conv_to<Row<double>>::from(ytrain3.row(ytrain3.n_rows - 5));
	ytrain3_p2_sol1 = conv_to<Row<double>>::from(ytrain3.row(ytrain3.n_rows - 6));
	ytrain3_p1_sol2 = conv_to<Row<double>>::from(ytrain3.row(ytrain3.n_rows - 7));
	ytrain3_p1_sol1 = conv_to<Row<double>>::from(ytrain3.row(ytrain3.n_rows - 8));



	Row<double> ytrain4_p1_sol1, ytrain4_p1_sol2, ytrain4_p2_sol1, ytrain4_p2_sol2, ytrain4_p3_sol1, ytrain4_p3_sol2;
	Row<double> ytrain4_diff_sol1, ytrain4_diff_sol2;//Variables for diffusion coefficient
	ytrain4_diff_sol2 = conv_to<Row<double>>::from(ytrain4.row(ytrain4.n_rows - 1));
	ytrain4_diff_sol1 = conv_to<Row<double>>::from(ytrain4.row(ytrain4.n_rows - 2));
	ytrain4_p3_sol2 = conv_to<Row<double>>::from(ytrain4.row(ytrain4.n_rows - 3));
	ytrain4_p3_sol1 = conv_to<Row<double>>::from(ytrain4.row(ytrain4.n_rows - 4));
	ytrain4_p2_sol2 = conv_to<Row<double>>::from(ytrain4.row(ytrain4.n_rows - 5));
	ytrain4_p2_sol1 = conv_to<Row<double>>::from(ytrain4.row(ytrain4.n_rows - 6));
	ytrain4_p1_sol2 = conv_to<Row<double>>::from(ytrain4.row(ytrain4.n_rows - 7));
	ytrain4_p1_sol1 = conv_to<Row<double>>::from(ytrain4.row(ytrain4.n_rows - 8));


	Row<double> ytrain5_p1_sol1, ytrain5_p1_sol2, ytrain5_p2_sol1, ytrain5_p2_sol2, ytrain5_p3_sol1, ytrain5_p3_sol2;
	Row<double> ytrain5_diff_sol1, ytrain5_diff_sol2;//Variables for diffusion coefficient
	ytrain5_diff_sol2 = conv_to<Row<double>>::from(ytrain5.row(ytrain5.n_rows - 1));
	ytrain5_diff_sol1 = conv_to<Row<double>>::from(ytrain5.row(ytrain5.n_rows - 2));
	ytrain5_p3_sol2 = conv_to<Row<double>>::from(ytrain5.row(ytrain5.n_rows - 3));
	ytrain5_p3_sol1 = conv_to<Row<double>>::from(ytrain5.row(ytrain5.n_rows - 4));
	ytrain5_p2_sol2 = conv_to<Row<double>>::from(ytrain5.row(ytrain5.n_rows - 5));
	ytrain5_p2_sol1 = conv_to<Row<double>>::from(ytrain5.row(ytrain5.n_rows - 6));
	ytrain5_p1_sol2 = conv_to<Row<double>>::from(ytrain5.row(ytrain5.n_rows - 7));
	ytrain5_p1_sol1 = conv_to<Row<double>>::from(ytrain5.row(ytrain5.n_rows - 8));



	int lentemp = Xtrain.n_cols;
	for (int i = 0; i < lentemp; i++) {
		//in this for loop we will save data from the phase diagram in array inside equations class.
		// Therefore, we can use them in different functions such as diffFunc.
		lst1_p1_sol1[i] = ytrain1_p1_sol1[i];
		lst1_p1_sol2[i] = ytrain1_p1_sol2[i];
		lst1_p2_sol1[i] = ytrain1_p2_sol1[i];
		lst1_p2_sol2[i] = ytrain1_p2_sol2[i];
		lst1_p3_sol1[i] = ytrain1_p3_sol1[i];
		lst1_p3_sol2[i] = ytrain1_p3_sol2[i];
		lst1_Diff_sol1[i] = ytrain1_diff_sol1[i];
		lst1_Diff_sol2[i] = ytrain1_diff_sol2[i];

		lst2_p1_sol1[i] = ytrain2_p1_sol1[i];
		lst2_p1_sol2[i] = ytrain2_p1_sol2[i];
		lst2_p2_sol1[i] = ytrain2_p2_sol1[i];
		lst2_p2_sol2[i] = ytrain2_p2_sol2[i];
		lst2_p3_sol1[i] = ytrain2_p3_sol1[i];
		lst2_p3_sol2[i] = ytrain2_p3_sol2[i];
		lst2_Diff_sol1[i] = ytrain2_diff_sol1[i];
		lst2_Diff_sol2[i] = ytrain2_diff_sol2[i];

		lst3_p1_sol1[i] = ytrain3_p1_sol1[i];
		lst3_p1_sol2[i] = ytrain3_p1_sol2[i];
		lst3_p2_sol1[i] = ytrain3_p2_sol1[i];
		lst3_p2_sol2[i] = ytrain3_p2_sol2[i];
		lst3_p3_sol1[i] = ytrain3_p3_sol1[i];
		lst3_p3_sol2[i] = ytrain3_p3_sol2[i];
		lst3_Diff_sol1[i] = ytrain3_diff_sol1[i];
		lst3_Diff_sol2[i] = ytrain3_diff_sol2[i];

		lst4_p1_sol1[i] = ytrain4_p1_sol1[i];
		lst4_p1_sol2[i] = ytrain4_p1_sol2[i];
		lst4_p2_sol1[i] = ytrain4_p2_sol1[i];
		lst4_p2_sol2[i] = ytrain4_p2_sol2[i];
		lst4_p3_sol1[i] = ytrain4_p3_sol1[i];
		lst4_p3_sol2[i] = ytrain4_p3_sol2[i];
		lst4_Diff_sol1[i] = ytrain4_diff_sol1[i];
		lst4_Diff_sol2[i] = ytrain4_diff_sol2[i];

		lst5_p1_sol1[i] = ytrain5_p1_sol1[i];
		lst5_p1_sol2[i] = ytrain5_p1_sol2[i];
		lst5_p2_sol1[i] = ytrain5_p2_sol1[i];
		lst5_p2_sol2[i] = ytrain5_p2_sol2[i];
		lst5_p3_sol1[i] = ytrain5_p3_sol1[i];
		lst5_p3_sol2[i] = ytrain5_p3_sol2[i];
		lst5_Diff_sol1[i] = ytrain5_diff_sol1[i];
		lst5_Diff_sol2[i] = ytrain5_diff_sol2[i];
	}
	// Use templates to specify that we want a NeighborSearch object which uses
	typedef NeighborSearch<NearestNeighborSort, EuclideanDistance, mat, KDTree> KNN;

	//KNN model(Xtrain, mlpack::neighbor::DUAL_TREE_MODE);
	KNN model(Xtrain, mlpack::neighbor::SINGLE_TREE_MODE);
	knnModel = model;
}

void equations::findNeighbor() {
	double p1_sol1 = 0, p1_sol2 = 0, p2_sol1 = 0, p2_sol2 = 0, p3_sol1 = 0, p3_sol2 = 0, diff_sol1 = 0, diff_sol2 = 0;
	double temp_p1_sol1, temp_p1_sol2, temp_p2_sol1, temp_p2_sol2, temp_p3_sol1, temp_p3_sol2, temp_diff_sol1, temp_diff_sol2;
	bool uniformAve = false;
	double weigth = 0, weigthTotal = 0;
	int index = 0;


	double t1 = tem - 273;
	double t2 = t1 * t1;
	double t3 = t2 * t1;
	double t4 = t3 * t1;
	arma::Mat<size_t> nei_Index;
	arma::mat nei_Distance;
	arma::mat sample;
	auto start = chrono::system_clock::now();

	/* As input varaibles for the knn we have temperature, cbar1 and cbar2.
	 The data have been devided into 5 subsections. The cbar values are the same in these input, but the temperature
	 are different. For the variation of temperature we have:
	 Section1 --> T(C°): [25, 125],
	 Section2 --> T(C°): [125, 225]
	 Section3 --> T(C°): [225, 325]
	 Section4 --> T(C°): [325, 425]
	 Section5 --> T(C°): [425, 525]

	 Instead of searching in 5 different files, by converting temperature in the range of [25, 125],
	 we can have only one file for searching.
	*/

	double temknn = t1;
	int flagknn = 1;
	if (temknn > 425) {
		temknn -= 400;
		flagknn = 5;
	}
	else if (temknn > 325) {
		temknn -= 300;
		flagknn = 4;
	}
	else if (temknn > 225) {
		temknn -= 200;
		flagknn = 3;
	}
	else if (temknn > 125) {
		temknn -= 100;
		flagknn = 2;
	}

	// There are three input parameters which are saved in temknn, cbLst[0], and cbLst[1]
	// We prepare the input variable and set it in sample:
	int k = 2;// indicator for k_neighbor
	sample << temknn << endr << cbLst[0] << endr << cbLst[1];
	knnModel.Search(sample, k, nei_Index, nei_Distance);

	for (int j = 0; j < k; j++) {
		// In this for loop, the obtained value of for each neighbor will be added to the coresponding variables.
		// In this test, k-neighbor is considered as 2.
		index = nei_Index[j];
		setTempValues(flagknn, index, temp_p1_sol1, temp_p1_sol2, temp_p2_sol1, temp_p2_sol2, temp_p3_sol1, temp_p3_sol2, temp_diff_sol1, temp_diff_sol2);
		if (uniformAve == true) {
			p1_sol1 += temp_p1_sol1;
			p1_sol2 += temp_p1_sol2;
			p2_sol1 += temp_p2_sol1;
			p2_sol2 += temp_p2_sol2;
			p3_sol1 += temp_p3_sol1;
			p3_sol2 += temp_p3_sol2;
			diff_sol1 += temp_diff_sol1;
			diff_sol2 += temp_diff_sol2;
			weigthTotal = k;
		}
		else {
			weigth = 1 / nei_Distance[j];
			weigthTotal += weigth;
			p1_sol1 += temp_p1_sol1 * weigth;
			p1_sol2 += temp_p1_sol2 * weigth;
			p2_sol1 += temp_p2_sol1 * weigth;
			p2_sol2 += temp_p2_sol2 * weigth;
			p3_sol1 += temp_p3_sol1 * weigth;
			p3_sol2 += temp_p3_sol2 * weigth;
			diff_sol1 += temp_diff_sol1 * weigth;
			diff_sol2 += temp_diff_sol2 * weigth;
		}
	}

	p1_sol1 /= weigthTotal;
	p1_sol2 /= weigthTotal;
	p2_sol1 /= weigthTotal;
	p2_sol2 /= weigthTotal;
	p3_sol1 /= weigthTotal;
	p3_sol2 /= weigthTotal;
	diff_sol1 /= weigthTotal;
	diff_sol2 /= weigthTotal;

	//The results are saved in resLst.
	resLst[0] = p1_sol1;
	resLst[1] = p1_sol2;
	resLst[2] = p2_sol1;
	resLst[3] = p2_sol2;
	resLst[4] = p3_sol1;
	resLst[5] = p3_sol2;
	resLst[6] = diff_sol1;
	resLst[7] = diff_sol2;
	

    // Writing result in a text file:
	knnfile << scientific << setprecision(9)
		<< t1 << "\t" << cbLst[0] << "\t" << cbLst[1] << "\t"
		<< p1_sol1 << "\t" << p1_sol2 << "\t" << p2_sol1 << "\t"
		<< p2_sol2 << "\t" << p3_sol1 << "\t" << p3_sol2 << "\t"
		<< diff_sol1 << "\t" << diff_sol2 << "\n";


	/*
	To make a comparison between KNN model and thermodynamics database, we can make a script. This script can be run
	in MatCalc. Therefore, we can make a comparison.
	The following lines will be used to save a script to obtain exact value of solubility based on the
	dynamic cbar of the system.
	*/
	matScript << "\nenter-composition x mg = " << setprecision(9) << cbLst[0] << " si = " << cbLst[1];
	matScript << "\nset-temperature-celsius " << setprecision(9) << t1;
	matScript << "\nset-automatic-startvalues" << "\ncalculate-equilibrium";
	matScript << "\nadd-table-entry cBar_data " << setprecision(9) << cbLst[0] << " " << cbLst[1];
	matScript << "\nadd-table-entry solubility_data  cEq_Sol1 cEq_Sol2";
	matScript << "\nadd-table-entry diffusion_data  Diff_Sol1 Diff_Sol2";
	matScript << "\nadd-table-entry T_data temValue cEq_Sol1\n";
}

void equations::setTempValues(int flag, int rowId, double& temp1, double& temp2, double& temp3, double& temp4, double& temp5, double& temp6, double& temp7, double& temp8) {
	if (flag == 1) {
		temp1 = lst1_p1_sol1[rowId];
		temp2 = lst1_p1_sol2[rowId];
		temp3 = lst1_p2_sol1[rowId];
		temp4 = lst1_p2_sol2[rowId];
		temp5 = lst1_p3_sol1[rowId];
		temp6 = lst1_p3_sol2[rowId];
		temp7 = lst1_Diff_sol1[rowId];
		temp8 = lst1_Diff_sol2[rowId];
	}

	else if (flag == 2) {
		temp1 = lst2_p1_sol1[rowId];
		temp2 = lst2_p1_sol2[rowId];
		temp3 = lst2_p2_sol1[rowId];
		temp4 = lst2_p2_sol2[rowId];
		temp5 = lst2_p3_sol1[rowId];
		temp6 = lst2_p3_sol2[rowId];
		temp7 = lst2_Diff_sol1[rowId];
		temp8 = lst2_Diff_sol2[rowId];
	}

	else if (flag == 3) {
		temp1 = lst3_p1_sol1[rowId];
		temp2 = lst3_p1_sol2[rowId];
		temp3 = lst3_p2_sol1[rowId];
		temp4 = lst3_p2_sol2[rowId];
		temp5 = lst3_p3_sol1[rowId];
		temp6 = lst3_p3_sol2[rowId];
		temp7 = lst3_Diff_sol1[rowId];
		temp8 = lst3_Diff_sol2[rowId];
	}

	else if (flag == 4) {
		temp1 = lst4_p1_sol1[rowId];
		temp2 = lst4_p1_sol2[rowId];
		temp3 = lst4_p2_sol1[rowId];
		temp4 = lst4_p2_sol2[rowId];
		temp5 = lst4_p3_sol1[rowId];
		temp6 = lst4_p3_sol2[rowId];
		temp7 = lst4_Diff_sol1[rowId];
		temp8 = lst4_Diff_sol2[rowId];
	}

	else if (flag == 5) {
		temp1 = lst5_p1_sol1[rowId];
		temp2 = lst5_p1_sol2[rowId];
		temp3 = lst5_p2_sol1[rowId];
		temp4 = lst5_p2_sol2[rowId];
		temp5 = lst5_p3_sol1[rowId];
		temp6 = lst5_p3_sol2[rowId];
		temp7 = lst5_Diff_sol1[rowId];
		temp8 = lst5_Diff_sol2[rowId];
	}
}