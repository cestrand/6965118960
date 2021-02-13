package projekt.essaerae;
import svm.SVMModel;
import svm.SVMParameter;
import svm.SVMProblem;
import weka.Matrix;
import weka.SiecB;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

import java.util.Random;

public class AnalizaGlass {

	public static void main(String[] args) throws Exception {
		Instances dane = DataSource.read("zasoby/weka-data/glass.arff");
		dane.setClassIndex(9);
//		wypiszDane(dane);

//		wykonajSiecBK2(dane, 2);
		/*Bayes Network Classifier
		not using ADTree
		#attributes=10 #classindex=9
		Network structure (nodes followed by parents)
		RI(3): Type
		Na(2): Type
		Mg(2): Type RI
		Al(3): Type RI
		Si(1): Type
		K(4): Type Mg
		Ca(4): Type Mg
		Ba(2): Type
		Fe(1): Type
		Type(6):
		LogScore Bayes: -1151.982994321905
		LogScore BDeu: -1657.559012637211
		LogScore MDL: -1596.9649336844057
		LogScore ENTROPY: -1181.1017925202127
		LogScore AIC: -1336.1017925202127

		61   6   3   0   0   0
		14  56   2   3   0   1
		8   2   7   0   0   0
		0   0   0  13   0   0
		0   1   0   0   8   0
		1   0   1   0   0  27*/


//		wykonajSiecBK2(dane, 5);
		/*Bayes Network Classifier
		not using ADTree
		#attributes=10 #classindex=9
		Network structure (nodes followed by parents)
		RI(3): Type
		Na(2): Type
		Mg(2): Type RI
		Al(3): Type RI
		Si(1): Type
		K(4): Type Mg Na
		Ca(4): Type Mg RI
		Ba(2): Type
		Fe(1): Type
		Type(6):
		LogScore Bayes: -1147.485351157255
		LogScore BDeu: -2207.2498097175176
		LogScore MDL: -1971.3316049959003
		LogScore ENTROPY: -1265.7057590205263
		LogScore AIC: -1528.705759020526

		61   6   3   0   0   0
		11  58   2   5   0   0
		8   2   7   0   0   0
		0   0   0  13   0   0
		0   1   0   0   8   0
		1   0   1   0   1  26*/

//		wykonajSiecBHC(dane, 2);
		/*Bayes Network Classifier
		not using ADTree
		#attributes=10 #classindex=9
		Network structure (nodes followed by parents)
		RI(3): Type Al
		Na(2): Type RI
		Mg(2): Type
		Al(3): Type Mg
		Si(1): Type
		K(4): Type RI
		Ca(4): Type K
		Ba(2): Type K
		Fe(1): Type
		Type(6):
		LogScore Bayes: -1178.7765507349482
		LogScore BDeu: -2065.8208423331184
		LogScore MDL: -1903.486172928757
		LogScore ENTROPY: -1262.2520391336466
		LogScore AIC: -1501.2520391336461

		61   6   3   0   0   0
		9  59   2   5   0   1
		8   2   7   0   0   0
		0   0   0  12   0   1
		0   0   0   0   9   0
		1   0   0   1   0  27*/

//		wykonajSiecBHC(dane, 5);
		/*Bayes Network Classifier
		not using ADTree
		#attributes=10 #classindex=9
		Network structure (nodes followed by parents)
		RI(3): Type Al Ca Mg
		Na(2): Type RI
		Mg(2): Type
		Al(3): Type Mg
		Si(1): Type
		K(4): Type RI Ca Al
		Ca(4): Type
		Ba(2): Type K Mg
		Fe(1): Type
		Type(6):
		LogScore Bayes: -1187.5969582964126
		LogScore BDeu: -7417.495957860137
		LogScore MDL: -4812.099676907575
		LogScore ENTROPY: -1981.5473289835613
		LogScore AIC: -3036.547328983568

		61   6   3   0   0   0
		9  62   2   3   0   0
		8   2   7   0   0   0
		0   0   0  13   0   0
		0   0   0   0   9   0
		1   1   0   0   0  27*/





//		wykonajMultilayerPerceptron(dane);
//		wykonajSVM(dane);
		// cross-validacje z macierzami pomyłek
	}

	private static void wykonajMultilayerPerceptron(Instances dane) throws Exception {
		MultilayerPerceptron mp = new MultilayerPerceptron();
		mp.setHiddenLayers("5,3");
		mp.setLearningRate(0.3);
		//mp.setGUI(true);
		mp.buildClassifier(dane);
		System.out.println(mp);
		Evaluation ev = new Evaluation(dane);
		ev.crossValidateModel(mp, dane, 30, new Random(123));
		System.out.println(ev.toMatrixString());
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

	private static void wykonajSiecBHCReverse(Instances dane, int maxNrOfParents) throws Exception {
		SiecB bn = SiecB.createHillClimbGlobal(maxNrOfParents, true);
		bn.buildClassifier(dane);
		System.out.println(bn);
		Matrix.show(bn.confMatrix(dane));
	}

	private static void wykonajSiecBHC(Instances dane, int maxNrOfParents) throws Exception {
		SiecB bn = SiecB.createHillClimbGlobal(maxNrOfParents);
		bn.buildClassifier(dane);
		System.out.println(bn);
		Matrix.show(bn.confMatrix(dane));
	}

	private static void wykonajSiecBK2(Instances dane, int maxNrOfParents) throws Exception {
		// sieć bayesowska k2
		SiecB bn = SiecB.createK2(maxNrOfParents);
		bn.buildClassifier(dane);
		System.out.println(bn);
		Matrix.show(bn.confMatrix(dane));
	}

}
