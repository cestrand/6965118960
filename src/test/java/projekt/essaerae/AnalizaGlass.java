package projekt.essaerae;
import svm.SVMModel;
import svm.SVMParameter;
import svm.SVMProblem;
import weka.Matrix;
import weka.SiecB;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

import java.util.Random;

public class AnalizaGlass {

	public static void main(String[] args) throws Exception {
		Instances dane = DataSource.read("zasoby/weka-data/glass.arff");
		dane.setClass(dane.attribute("Type"));
//		wypiszDane(dane);

//		wykonajSiecBK2P2(dane);

//		wykonajSiceBHCGP2(dane);

//		wykonajSiecBHCLP2(dane);

//		wykonajSiecBHCGP5(dane);

//		wykonajSiecBHCLP5(dane);

		wykonajSVMC(dane);

//		wykonajSVMNu(dane);


	}

	private static void wykonajSiecBHCLP5(Instances dane) throws Exception {
		int maxNrOfParents = 5;
		SiecB bn = SiecB.createHillClimbLocal(maxNrOfParents);
		bn.buildClassifier(dane);
		System.out.println("Sieć Bayesowska Hill Climbing (local) max rodziców 5:");
		System.out.println(bn);

		int[][] confM = bn.confMatrix(dane);
		Matrix.show(confM);
		pokazDokladnosc(dane, confM);

		System.out.println("Walidacja krzyżowa:");
		int[][] confMVal = bn.confMatrixOfXVal(dane, 10, new Random(3));
		Matrix.show(confMVal);
		pokazDokladnosc(dane, confMVal);
	}

	private static void wykonajSiecBHCLP2(Instances dane) throws Exception {
		int maxNrOfParents = 2;
		SiecB bn = SiecB.createHillClimbLocal(maxNrOfParents);
		bn.buildClassifier(dane);
		System.out.println("Sieć Bayesowska Hill Climbing (local) max rodziców 2:");
		System.out.println(bn);

		int[][] confM = bn.confMatrix(dane);
		Matrix.show(confM);
		pokazDokladnosc(dane, confM);
	}

	private static void wykonajSiceBHCGP2(Instances dane) throws Exception {
		int maxNrOfParents = 2;
		SiecB bn = SiecB.createHillClimbGlobal(maxNrOfParents);
		System.out.println("Sieć Bayesowska Hill Climbing (global) max rodziców 2:");
		bn.buildClassifier(dane);
		System.out.println(bn);

		int[][] confM = bn.confMatrix(dane);
		Matrix.show(confM);
		pokazDokladnosc(dane, confM);
	}

	private static void wykonajSiecBHCGP5(Instances dane) throws Exception {
		int maxNrOfParents = 5;
		SiecB bn = SiecB.createHillClimbGlobal(maxNrOfParents);
		bn.buildClassifier(dane);
		System.out.println("Sieć Bayesowska Hill Climbing (global) max rodziców 5:");
		System.out.println(bn);

		int[][] confM = bn.confMatrix(dane);
		Matrix.show(confM);
		pokazDokladnosc(dane, confM);

		System.out.println("Walidacja krzyżowa:");
		int[][] confMVal = bn.confMatrixOfXVal(dane, 10, new Random(3));
		Matrix.show(confMVal);
		pokazDokladnosc(dane, confMVal);
	}

	private static void wykonajSiecBK2P2(Instances dane) throws Exception {
		int maxNrOfParents=2;
		// sieć bayesowska k2
		SiecB bn = SiecB.createK2(maxNrOfParents);
		bn.buildClassifier(dane);
		System.out.println("Sieć Bayesowska K2 max rodziców 2:");
		System.out.println(bn);

		int[][] confM = bn.confMatrix(dane);
		Matrix.show(confM);
		pokazDokladnosc(dane, confM);
	}


	private static void wykonajSVMC(Instances dane) throws Exception {
		Standardize flt = new Standardize();
		flt.setInputFormat(dane);
		Instances danestd = Filter.useFilter(dane, flt);

		SVMProblem problem = SVMProblem.fromInstances(danestd);
		problem.par.svm_type = SVMParameter.C_SVC;
		problem.par.kernel_type = SVMParameter.RBF;
		problem.par.C = 16; //sprawdzone: 10, 15, 16, 18, 20
		problem.par.gamma = 1./(danestd.numAttributes()-1);
		SVMModel model= problem.train();

		int[][] M = problem.confMatrix(model);
		Matrix.show(M);
		pokazDokladnosc(danestd, M);

		System.out.println("Walidacja krzyżowa:");
		double[] celX = new double[danestd.numInstances()];
		double dokladnosc = problem.crossValidation(problem.par, 10, celX);
		System.out.printf("Dokładność:\t%.2f\n", dokladnosc*100);
	}

	private static void wykonajSVMNu(Instances dane) throws Exception {
		Standardize flt = new Standardize();
		flt.setInputFormat(dane);
		Instances danestd = Filter.useFilter(dane, flt);

		SVMProblem problem = SVMProblem.fromInstances(danestd);
		problem.par.svm_type = SVMParameter.NU_SVC;
		problem.par.nu = 0.25;
		double[] celX = new double[dane.numInstances()];
		SVMParameter par = problem.par;

		double[] g = {0.11};//0.1, 0.11, 0.12, 0.15, 0.18, 1./(danestd.numAttributes()-1)};
		for (double gamma:g) {
			par.gamma = gamma;
			double dokladnosc = problem.crossValidation(problem.par, 10, celX);
			System.out.printf("Dokładność:\t%.2f\n", dokladnosc*100);
			System.out.println("Dla gamma = "+gamma);
		}

		problem.par.gamma = 0.11;
		SVMModel model = problem.train();
		int[][] M = problem.confMatrix(model);
		Matrix.show(M);
		pokazDokladnosc(dane, M);
	}

	private static void wypiszDane(Instances dane) {
		System.out.println(dane);
	}

	public static void pokazDokladnosc(Instances dane, int[][] M){
		int N = M.length;
		int poprawne = 0;
		double wszystkie = dane.numInstances();
		for (int i = 0; i < N; i++) {
			poprawne += M[i][i];
		}
		int bledne = (int)wszystkie - poprawne;
		System.out.printf("Poprawne:\t%d --- %.2f%%\n", poprawne, poprawne / wszystkie * 100);
		System.out.printf("Błędne:  \t%d --- %.2f%%\n", bledne, bledne / wszystkie * 100);
	}



}
