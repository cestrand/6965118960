package projekt.essaerae;
import weka.Matrix;
import weka.SiecB;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class AnalizaGlass {

	public static void main(String[] args) throws Exception {
		Instances dane = DataSource.read("zasoby/weka-data/glass.arff");
		dane.setClassIndex(9);
		wypiszDane(dane);

		wykonajSiecBK2(dane, 2);
//		Bayes Network Classifier
//		not using ADTree
//		#attributes=10 #classindex=9
//		Network structure (nodes followed by parents)
//		RI(3): Type
//		Na(2): Type
//		Mg(2): Type RI
//		Al(3): Type RI
//		Si(1): Type
//		K(4): Type Mg
//		Ca(4): Type Mg
//		Ba(2): Type
//		Fe(1): Type
//		Type(7):
//		LogScore Bayes: -1154.1645405405316
//		LogScore BDeu: -1772.6151459308992
//		LogScore MDL: -1691.4074842654582
//		LogScore ENTROPY: -1205.7866549059813
//		LogScore AIC: -1386.7866549059809

// 		61   6   3   0   0   0   0
//		14  56   2   0   3   0   1
//		8   2   7   0   0   0   0
//		0   0   0   0   0   0   0
//		0   0   0   0  13   0   0
//		0   1   0   0   0   8   0
//		1   0   1   0   0   0  27

		wykonajSiecBK2(dane, 5);
//		Bayes Network Classifier
//		not using ADTree
//      #attributes=10 #classindex=9
//		Network structure (nodes followed by parents)
//		RI(3): Type
//		Na(2): Type
//		Mg(2): Type RI
//		Al(3): Type RI
//		Si(1): Type
//		K(4): Type Mg Na
//		Ca(4): Type Mg RI
//		Ba(2): Type
//		Fe(1): Type
//		Type(7):
//		LogScore Bayes: -1149.6668973758815
//		LogScore BDeu: -2434.0768119165823
//		LogScore MDL: -2130.703472045588
//		LogScore ENTROPY: -1307.0261537397337
//		LogScore AIC: -1614.0261537397332

//		61   6   3   0   0   0   0
//		11  58   2   0   5   0   0
//		8   2   7   0   0   0   0
//		0   0   0   0   0   0   0
//		0   0   0   0  13   0   0
//		0   1   0   0   0   8   0
//		1   0   1   0   0   1  26


//		wykonajSiecBHC(dane, 2);
//		wykonajSiecBHCReverse(dane, 2);

		// multilayer perceptron
		wykonajSVM(dane);
		// cross-validacje z macierzami pomyłek
	}

	private static void wykonajSVM(Instances dane) {

	}

	private static void wypiszDane(Instances dane) {
		System.out.println(dane);
	}

	private static void wykonajSiecBHCReverse(Instances dane, int maxNrOfParents) throws Exception {
		SiecB bn = SiecB.createHillClimbGlobal(maxNrOfParents, true);
		bn.buildClassifier(dane);
		System.out.println(bn);
	}

	private static void wykonajSiecBHC(Instances dane, int maxNrOfParents) throws Exception {
		SiecB bn = SiecB.createHillClimbGlobal(maxNrOfParents);
		bn.buildClassifier(dane);
		System.out.println(bn);
	}

	private static void wykonajSiecBK2(Instances dane, int maxNrOfParents) throws Exception {
		// sieć bayesowska k2
		SiecB bn = SiecB.createK2(maxNrOfParents);
		bn.buildClassifier(dane);
		System.out.println(bn);
		Matrix.show(bn.confMatrix(dane));
	}

}
