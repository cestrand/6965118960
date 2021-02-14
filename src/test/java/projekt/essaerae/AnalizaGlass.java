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

public class AnalizaGlass {

	public static void main(String[] args) throws Exception {
		Instances dane = DataSource.read("zasoby/weka-data/glass.arff");
		dane.setClassIndex(9);
//		wypiszDane(dane);

		wykonajSiecBK2(dane, 2);

//		wykonajSiecBHCG(dane, 2);

		wykonajSiecBHCG(dane, 5);

//		wykonajSiecBHCL(dane, 2);

		wykonajSiecBHCL(dane, 5);

//		wykonajSVM(dane);


	}


	private static void wykonajSVM(Instances dane) throws Exception {
		Standardize flt = new Standardize();
		flt.setInputFormat(dane);
		Instances danestd = Filter.useFilter(dane, flt);

		SVMProblem problem = SVMProblem.fromInstances(dane);
		problem.par.svm_type = SVMParameter.C_SVC;
		problem.par.nu = 0.2;
		problem.par.gamma = 0.15;
		SVMModel model = problem.train();
		Matrix.show(problem.confMatrix(model));
	}

	private static void wypiszDane(Instances dane) {
		System.out.println(dane);
	}

	private static void wykonajSiecBHCG(Instances dane, int maxNrOfParents) throws Exception {
		SiecB bn = SiecB.createHillClimbGlobal(maxNrOfParents);
		bn.buildClassifier(dane);
		System.out.println(bn);
		Matrix.show(bn.confMatrix(dane));
		pokazDokladnosc(dane, bn);
	}

	private static void wykonajSiecBHCL(Instances dane, int maxNrOfParents) throws Exception {
		SiecB bn = SiecB.createHillClimbLocal(maxNrOfParents);
		bn.buildClassifier(dane);
		System.out.println(bn);
		Matrix.show(bn.confMatrix(dane));
		pokazDokladnosc(dane, bn);
	}

	private static void wykonajSiecBK2(Instances dane, int maxNrOfParents) throws Exception {
		// sieć bayesowska k2
		SiecB bn = SiecB.createK2(maxNrOfParents);
		bn.buildClassifier(dane);
		System.out.println(bn);
		Matrix.show(bn.confMatrix(dane));
		pokazDokladnosc(dane, bn);
	}

	private static void pokazDokladnosc(Instances dane, SiecB bn) throws Exception {
		Evaluation ev = new Evaluation(dane);
		ev.evaluateModel(bn, dane);
		System.out.printf("Poprawne:\t%d --- %.2f%%\n", (int) ev.correct(), ev.correct() / ev.numInstances() * 100);
		System.out.printf("Błędne:  \t%d --- %.2f%%\n", (int) ev.incorrect(), ev.incorrect() / ev.numInstances() * 100);
	}

}
