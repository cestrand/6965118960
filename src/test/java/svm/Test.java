package svm;


import java.util.Random;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class Test {

	public static void main(String[] args) throws Exception {
//2		
		Instances dane = DataSource.read("zasoby/weka-data/diabetes.arff");
		dane.setClassIndex(dane.numAttributes()-1);
		Standardize flt = new Standardize();
		flt.setInputFormat(dane);
		dane = Filter.useFilter(dane, flt);				// uwaga na FILTR		
//3
		System.out.println("A teraz tworzymy Problem:");
		SVMProblem prDia = SVMProblem.fromInstances(dane); 		//dane po  Standardyzacji
//		prDia.par.kernel_type = Parameter.POLY; 		//domy�lnie: Parameter.RBF
//		prDia.par.degree = 3;							//domy�lnie =3
		System.out.println("Pora na trening:");
		// ju� jest - zapisane w pliku
		SVMModel model = prDia.train();   				//to powstaje klasyfikator
		
		//lub - nieco og�lniej
//			SVMParameter par = new SVMParameter();
//			par.C = 10;
//			par.gamma = 0.3;
//			model = prDia.train(par);
		//... i testy...
		model.saveTo("zasoby/diab-model-POLY3.mod");			//UWAGA - tylko w po��czeniu z filtrem
//		svm_model model = svm.svm_load_model("diab-model-Poly2.mod"); 
		//test wiersza nr np. 3
		System.out.println("Po stand.: "+dane.instance(4));
//		double dec = model.predict((SVMNode[]) prDia.x[4]); //predykcj� wykonuje teraz klasa Model
		double dec = model.predict(dane.instance(4)); 			//lub - dla ci�gu liczb
		dec = model.predict(dane.instance(4).toDoubleArray()); 
		
		System.out.println("decyzja ="+dec);
		System.out.println(".a by�o: "+prDia.y[4]);
		dec = model.predict(dane.instance(333));
		System.out.println("decyzja ="+dec);
		System.out.println(".a by�o: "+prDia.y[333]);
		//Wyznaczy� macierz b��d�w
		
		int m = dane.numClasses();
		int[][] M = new int[m][m];		// GDZIE to zapisa�? -> raczej w SVMProblem - zna dane
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
		double[] w = new double[prDia.getNumberOfTrainingAttributes()];
		for(int i=0; i<w.length; i++)
			w[i] = los.nextGaussian();
//		double pred = model.predict(w); lub - po zamianie na Instance
		Instance wI = new DenseInstance(1, w);
		double pred = model.predict(wI);
	}

}
