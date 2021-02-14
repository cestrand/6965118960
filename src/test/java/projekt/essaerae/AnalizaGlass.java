package projekt.essaerae;
import svm.SVMModel;
import svm.SVMParameter;
import svm.SVMProblem;
import weka.Matrix;
import weka.SiecB;
import weka.classifiers.Evaluation;
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

		wykonajSVM(dane);


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


	private static void wykonajSVM(Instances dane) throws Exception {
		Standardize flt = new Standardize();
		flt.setInputFormat(dane);
		Instances danestd = Filter.useFilter(dane, flt);

		SVMProblem problem = SVMProblem.fromInstances(dane);
		problem.par.svm_type = SVMParameter.C_SVC;
		problem.par.kernel_type = SVMParameter.RBF;
		problem.par.C = 10;
		problem.par.gamma = 0.5;
		SVMModel model= problem.train();

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
