package svm;

import java.util.Random;

import weka.classifiers.functions.SMO;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.LibSVMLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;
import weka.core.converters.ConverterUtils.DataSource;

public class Test2 {

	public static void main(String[] args) throws Exception {
//1
		LibSVMLoader svml = new LibSVMLoader();
		svml.setURL("file:zasoby/sonar_scale.txt");
		SVMProblem pr = new SVMProblem("file:zasoby/sonar_scale.txt");
		System.out.println("Liczba klas = " + pr.classNum);
		pr.train();
//		= svml.getDataSet();
//		// kt�ry atrybut jest decyzyjny? 
//		// rozpozna� struktur�...
//		System.out.println(dane.instance(122));
//		System.out.println(dane.instance(122).getClass());
//		dane.setClassIndex(dane.numAttributes()-1);
//		System.out.println(dane.classAttribute().numValues());
//		dane.classAttribute();
//2		
		Instances dane; 
		dane = DataSource.read("zasoby/weka-data/diabetes.arff");
		dane.setClassIndex(dane.numAttributes()-1);
		Standardize flt = new Standardize();
//		Normalize flt = new Normalize();
		flt.setInputFormat(dane);
		dane = Filter.useFilter(dane, flt);
//		System.out.println(".............................................");
//		System.out.println(flt.toSource("Flt", dane));
//		System.out.println(".............................................");
//		SMO smo = new SMO();
//		smo.buildClassifier(dane);
//		System.out.println(smo);
		
		// sprawdzamy - w Wece!
//		smo.setKernel(SMO.???);
		
//3
		System.out.println("A teraz tworzymy Problem:");
		SVMProblem prDia = new SVMProblem(dane); 		//dane po  Standardyzacji
//		prDia.par.kernel_type = Parameter.POLY; //domy�lnie: Parameter.RBF
//		prDia.par.degree = 3;	//domy�lnie =3
		prDia.par.C = 22.5; //domy�lnie =1

		System.out.println("Pora na trening:");
		// ju� jest - zapisane w pliku
		SVMModel model = prDia.train();   //to b�dzie klasyfikator
		//... i testy...
		model.saveTo("zasoby/diab-model-POLY3.mod");
//		svm_model model = svm.svm_load_model("diab-model-Poly2.mod");
		//test wiersza nr np. 3
		System.out.println("Po stand.: "+dane.instance(4));
		double dec = model.predict((SVMNode[]) prDia.x[4]);  //predykcj� wykonuje teraz klasa Model
		System.out.println("decyzja ="+dec);
		System.out.println(".a by�o: "+prDia.y[4]);
		dec = model.predict((SVMNode[]) prDia.x[333]);
		System.out.println("decyzja ="+dec);
		System.out.println(".a by�o: "+prDia.y[333]);
		//Wyznaczy� macierz b��d�w
		
		int m = dane.numClasses();
		int[][] M = new int[m][m];
		for(int i=0; i<prDia.l; i++) {
			double pred = model.predict((SVMNode[]) prDia.x[i]); 
			M[(int)prDia.y[i]][(int)pred]++; //macierz b��d�w
		}
		System.out.println("Macierz b��d�w");
		for(int i=0; i<m; i++) {  //i-ty wiersz
			for(int j=0; j<m; j++)
				System.out.printf("%3d ",M[i][j]);
			System.out.println();
		}	
		//dla danych losowych...
		Random los = new Random();
		double[] w = new double[prDia.M];
		for(int i=0; i<w.length; i++)
			w[i] = los.nextGaussian();
//		double pred = model.predict(w); lub - po zamianie na Instance
		Instance wI = new DenseInstance(1, w);
		double pred = model.predict(wI);
		
		
//		int[][] lolololo = prDia.confMatrix2(dane, model);
//		int[][] lolololo2 = SVMProblem.confMatrix(dane,prDia ,model);
//		SVMProblem.show(lolololo);
//		SVMProblem.show(lolololo2);
	}
}
