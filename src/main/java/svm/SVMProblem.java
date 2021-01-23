package svm;


import java.util.TreeSet;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_problem;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.LibSVMLoader;

public class SVMProblem extends svm_problem{
	SVMParameter par = new SVMParameter(); //domy�lnie - C-SVC: klasyfikacja z param. C=1	
	SVMModel model;
	int M;	//liczba atrybut�w (bez dec.)
	int classNum;
	
	public SVMProblem(String nazwaPl) throws Exception {	//format danych: libSVM	
		LibSVMLoader svml = new LibSVMLoader();
		svml.setURL(nazwaPl);
		Instances dane = svml.getDataSet();
		//itd - dane ju� s�, co to za problem?
		// kt�ry atrybut jest decyzyjny?  - Ostatni! - ale jako NUMERIC(!?)
		// rozpozna� struktur�... - to jest SparseInstance
		int iDec = dane.numAttributes()-1;
		dane.setClassIndex(iDec);
		y = dane.attributeToDoubleArray(iDec); //tu ju� s� decyzje
		int m = dane.numAttributes(); 		//jeden do pomini�cia
		int n = dane.numInstances();
		this.l = n;
		this.M = m-1;   					//bez decyzyjnego
		x = new SVMNode[l][M];
		TreeSet<Double> ts =new TreeSet();
		for(int i=0; i<n; i++) {
			Instance wiersz = dane.instance(i);
			// lista - ziera� warto�ci (int)(this.y[i]) - ile jest r�nych
			// -> classNum
			// je�li > 16 - to znaczy, �e =1 (SVR lub 1-SVM)
			ts.add(wiersz.classValue());
			for(int j=0; j<iDec; j++) 		//!! Pomin�� kolumn� iDec - decyzyjn�
				x[i][j] = new SVMNode(j+1,wiersz.value(j));
		}
		par.gamma = 1./M; //korekta warto�ci domy�lnej
		classNum = 1;
		if (ts.size() < 20)
			classNum = ts.size();
		
	}
	
	// do��czy� wariant z domy�ln� standaryzacj� lub normalizacj�
	// - raczej NIE - przepu�ci� przez filtr Standardize lub Normalize
	// - czy (i gdzie) zapami�ta� u�yty filtr?
	
	public SVMProblem(Instances dane) {   	//dane - jeden z atryb. jest KLASOWY
			//odczyta� i utworzy� y[] oraz x[][]
		int iDec = dane.classIndex();   //chyba ostatni? a mo�e pierwszy?
		if (iDec<0)						//jesli BRAK - wskazujemy ostatni
			dane.setClassIndex(dane.numAttributes()-1);
		y = dane.attributeToDoubleArray(iDec); //tu ju� s� decyzje
		int m = dane.numAttributes(); 	//jeden do pomini�cia
		int n = dane.numInstances();
		this.classNum = dane.numClasses(); //wiadomo z g�ry
		this.l = n;
		this.M = m-1;   				//bez decyzyjnego
		x = new SVMNode[l][M];
		for(int i=0; i<n; i++) {
			Instance wiersz = dane.instance(i);
			for(int j=0; j<iDec; j++) 	//!! Pomin�� kolumn� iDec - decyzyjn�
				x[i][j] = new SVMNode(j+1,wiersz.value(j));
			//do pomini�cia - je�li iDec=M = m-1
			for(int j=iDec+1; j<m; j++) //!! Pomin�� kolumn� iDec - decyzyjn� 
				x[i][j] = new SVMNode(j,wiersz.value(j));
		}
		par.gamma = 1./M; //korekta warto�ci domy�lnej
	}
	
	
	public SVMParameter getPar() {
		return par;
	}

	public int getM() {
		return M;
	}

	public SVMModel train() {
		//svm_model model = svm.svm_train(this, par);
		//this.classNum = model.nr_class;
		model =new SVMModel(svm.svm_train(this, par));
		return model; 	//dla SWOICH parametr�w 
	}
	public SVMModel train(SVMParameter par) {
		return new SVMModel(svm.svm_train(this, par));	//par - tym razem z zewn�trz
	}

	public static int[][] confMatrix(Instances dane, SVMProblem prDia, SVMModel model){
		int m = dane.numClasses();
		int[][] M = new int[m][m];
		for(int i=0; i<prDia.l; i++) {
			double pred = model.predict((SVMNode[]) prDia.x[i]); 
			M[(int)prDia.y[i]][(int)pred]++;
		}
		return M;
	}
	
	
	public int[][] confMatrix2(Instances dane, SVMModel model){
		int m = dane.numClasses();
		int[][] M = new int[m][m];
		for(int i=0; i<l; i++) {
			double pred = model.predict((SVMNode[]) x[i]); 
			M[(int)y[i]][(int)pred]++;
		}
		return M;
	}
	
	static public void show(int[][] M) {
		int m = M.length;
		for (int i = 0; i < m; i++) { // i-ty wiersz
			for (int j = 0; j < m; j++)
				System.out.printf("%3d ", M[i][j]);
			System.out.println();
		}
	}
	
	public void confusionMatrix(SVMModel model) {
		int m = model.nr_class;
		int[][] M = new int[m][m];
		for(int i=0; i<this.l ; i++){
			double pred = model.predict((SVMNode[]) this.x[i]);
			M[(int)this.y[i]][(int)pred]++;
		}
		System.out.println("Macierz b��d�w: ");
		for(int i = 0; i<m;i++) {
			for(int j =0; j <m;j++)
				System.out.printf("%3d ", M[i][j]);
			System.out.println();
		}
	}
	
	public double crossValidation(SVMParameter par, int nr_fold, double[] target) {
		if(par.gamma ==0)
			par.gamma = 1./M;
		svm.svm_cross_validation(this, par, nr_fold, target);
		int licznikOK =0;
		int m = this.classNum;
		int[][] M = new int[m][m];
		for(int i=0; i<this.l ; i++){
			M[(int)this.y[i]][(int)target[i]]++;
			if(target[i]==y[i]) licznikOK++;
		}
		System.out.println("Macierz b��d�w dla x-walidacji");
		for(int i = 0; i<m;i++) {
			for(int j =0; j <m;j++)
				System.out.printf("%3d ", M[i][j]);
			System.out.println();
		}
		return (double)licznikOK/l;
	}
	
	
}
