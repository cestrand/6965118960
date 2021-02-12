package svm;

import java.io.IOException;

import weka.Matrix;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class Test1C {

	public static void main(String[] args) throws Exception {
		
		Instances dane;
		dane = DataSource.read("zasoby/weka-data/diabetes.arff");
		dane.setClassIndex(dane.numAttributes() - 1);
		
		Standardize flt = new Standardize();
		flt.setInputFormat(dane);
		dane = Filter.useFilter(dane,  flt);
		
		totalGroup(dane, 0.3);
		
		// Zadanie: dwie chmury typu 1-SVM - osobno dla chorych i zdrowych
		// - Stworzy� dwie chmury danych (rozdzieili� zdrowych i chorych)
		// czy elementy skrajne s� bardziej podatne na b��dy decyzji? 
		
	}

	private static void totalGroup(Instances dane, double skrajne) throws IOException {
		System.out.println("A teraz tworzymy Problem typu 1-SVM:");
		SVMProblem prDia1 = SVMProblem.fromInstances(dane);
		prDia1.par.svm_type = SVMParameter.ONE_CLASS;
		prDia1.par.nu = skrajne;
		prDia1.par.gamma = 0.15; // optymalna warto�� wyk�adnika w j�drze RBF
		
		SVMModel model1= prDia1.train();
		model1.saveTo("zasoby/diab-model-1-RBF.mod");
//		prDia.confusionMatrix(model); 	// robiona dla zagadnie� klasyfikacji, tutaj nie zadzia�a
		
		int plus = 0;
		int[][] M = new int[2][2];
		
		for(int i = 0; i < prDia1.l; i++) {
			double pred = model1.predict((SVMNode[]) prDia1.x[i]);
			if (pred > 0) {
				plus++;
			} else {
				pred = 0;
			}
			M[(int) prDia1.y[i]][(int) pred]++;
		}
		
		System.out.println("Rozk�ad w chmurze danych: " + plus + ":" + (prDia1.l - plus));
		System.out.println("Macierz konfiguracji");
		
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				System.out.printf("%3d ", M[i][j]);
			}
			System.out.println();
		}
		// czy elementy skrajne s� bardziej podatne na b��dy decyzji? 
		//1. Porownanie prDia1 z prDiaNu
		SVMProblem prDiaNu = SVMProblem.fromInstances(dane); 		//dane po  Standardyzacji
//		prDia.par.kernel_type = Parameter.POLY; //domy�lnie: Parameter.RBF
		prDiaNu.par.svm_type = SVMParameter.NU_SVC;
		prDiaNu.par.nu = 1./3; 		//mniej ni� 1/3 b��dnych decyzji, przynajmniej 1/3 SVN
		prDiaNu.par.gamma = 0.15;
		SVMModel modelNu = prDiaNu.train();   	//to b�dzie klasyfikator
		//lub - odczyt z pliku
		System.out.println("...........................................");
		modelNu = SVMModel.loadFrom("zasoby/diab-model-nu-RBF.mod");
		//System.out.println("Z pliku: "+modelNu);

		Matrix.show(prDiaNu.confMatrix(modelNu));
		
		int[][]Mskr = new int[2][2];
		int[][]Mstd = new int[2][2];
		for(int i = 0; i < prDia1.l; i++) {
			double pred = model1.predict((SVMNode[]) prDia1.x[i]);
			if (pred < 0) { // elem.skrajny
				double dec = modelNu.predict((SVMNode[]) prDiaNu.x[i]);
				Mskr[(int) prDiaNu.y[i]][(int) dec]++; //macierz bledow
			} else { //elem. standardowy
				double dec = modelNu.predict((SVMNode[]) prDiaNu.x[i]);
				Mstd[(int) prDiaNu.y[i]][(int) dec]++; //macierz bledow
			}
		}
		System.out.println("Macierz b��d�w dla skrajnych:");
		for (int i = 0; i < 2; i++) { //i-ty wiersz
			for (int j = 0; j < 2; j++) {
				System.out.printf("%3d ", Mskr[i][j]);
			}
			System.out.println();
		}
		System.out.println("Macierz b��d�w dla standardowych:");
		for (int i = 0; i < 2; i++) { //i-ty wiersz
			for (int j = 0; j < 2; j++) {
				System.out.printf("%3d ", Mstd[i][j]);
			}
			System.out.println();
		}
		System.out.println("Hipoteza: elementy skrajne (na poziomie <20%)"+
		"mog�y istotnie wp�yn�� na klasyfikator. ");
		
		//2. Porownanie prDia1 z prDiaNuStd - nauka bez udzia�u element�w skrajnych 
			// skrajne - usuwamy
		Instances daneStd = new Instances(dane);
		for(int i = prDia1.l-1; i >= 0; i--) {
			double pred = model1.predict((SVMNode[]) prDia1.x[i]);
			if (pred < 0)  // elem.skrajny
				daneStd.delete(i);
		}
		SVMProblem prDiaStd = SVMProblem.fromInstances(daneStd); 		//dane po  Standardyzacji
		prDiaStd.par.svm_type = SVMParameter.NU_SVC;
		prDiaStd.par.nu = 1./3; 		//mniej ni� 1/3 b��dnych decyzji, przynajmniej 1/3 SVN
		prDiaStd.par.gamma = 0.15;		// czy ten parametr jest optymalny (dla zmienionego zbioru)?
		SVMModel modelStd = prDiaStd.train();   //to b�dzie klasyfikator dla zwyk�ych
		
		Mskr = new int[2][2];
		Mstd = new int[2][2];
		System.out.println("Model dla elem. standardowych");

		Matrix.show(prDiaStd.confMatrix(modelStd));
		Matrix.show(prDiaNu.confMatrix(modelStd)); // nowy model dla starych danych (dla wiekszego zbioru)
		// tzn ze zbior uczacy to Std a testowy to caly czyli razem ze skrajnymi
		//Mamy ze niby bledow lacznie wiecej, ale dla standardowych mniej
		
		for(int i = 0; i < prDia1.l; i++) {
			double pred = model1.predict((SVMNode[]) prDia1.x[i]);
			if (pred < 0) { // elem.skrajny - odczytujemy z prDiaNu
				double dec = modelStd.predict((SVMNode[]) prDiaNu.x[i]); //elementy chcemy brac z oryginalnego zbioru
				Mskr[(int) prDiaNu.y[i]][(int) dec]++; //macierz bledow
			} 
		}
		System.out.println("Macierz b��d�w dla skrajnych:");
		for (int i = 0; i < 2; i++) { //i-ty wiersz
			for (int j = 0; j < 2; j++) {
				System.out.printf("%3d ", Mskr[i][j]);
			}
			System.out.println();
		}
	//	System.out.println("Macierz b��d�w dla standardowych:");
		///for (int i = 0; i < 2; i++) { //i-ty wiersz
		//	for (int j = 0; j < 2; j++) {
		//		System.out.printf("%3d ", Mstd[i][j]);
		//	}
		//	System.out.println();
	//	}
	
	
	}
}