package svm;


import java.util.TreeSet;

import libsvm.svm;
import libsvm.svm_problem;
import weka.Matrix;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.LibSVMLoader;

public class SVMProblem extends svm_problem{
	public SVMParameter par = new SVMParameter(); //domy�lnie - C-SVC: klasyfikacja z param. C=1
	SVMModel model;

	/**
	 * Liczba atrybutów w danych minus jeden, bo jeden jest decyzyjny.
	 */
	int numberOfTrainingAttributes;
	int classNum;

	public SVMProblem() {

	}

	/**
	 * Ładownaie modelu z pliku. Ostatnia kolumna ma być decyzyjna czyli ta, którą będziemy predykować.
	 * Można też użyć metody statycznej fromInstances i samemu załadować dane ustawiając kolumnę decyzyjną.
	 * @param nazwaPl Ścieżka do pliku z danymi o formacie ładowalnym przez Weka.
	 * @throws Exception
	 */
	public static SVMProblem fromFile(String nazwaPl) throws Exception {
		LibSVMLoader svml = new LibSVMLoader();
		svml.setURL(nazwaPl);
		Instances dane = svml.getDataSet();
		int iDec = dane.numAttributes()-1;
		dane.setClassIndex(iDec);
		return fromInstances(dane);
	}

	/**
	 *
	 * @return Liczba wierszy z danymi w tym problemie.
	 */
	public int numberOfInstances() {
		return this.l;
	}

	/**
	 * Ładuje dane wraz z informacjami o atrybucie klasowym z dane.
	 * @param dane
	 * @return
	 */
	public static SVMProblem fromInstances(Instances dane) {
		int iDec = dane.classIndex();
		if (iDec<0)
			dane.setClassIndex(dane.numAttributes()-1);
		SVMProblem p = new SVMProblem();

		p.y = dane.attributeToDoubleArray(iDec);
		int m = dane.numAttributes();
		int n = dane.numInstances();
		p.classNum = dane.numClasses();
		p.l = n;
		p.numberOfTrainingAttributes = m-1;
		p.x = new SVMNode[p.l][p.numberOfTrainingAttributes];

		TreeSet<Double> ts =new TreeSet();
		for(int i=0; i<n; i++) {
			Instance wiersz = dane.instance(i);
			ts.add(wiersz.classValue());
			for(int j=0; j<iDec; j++)
				p.x[i][j] = new SVMNode(j+1,wiersz.value(j));
			for(int j=iDec+1; j<m; j++)
				p.x[i][j] = new SVMNode(j,wiersz.value(j));
		}

		p.par.gamma = 1./ p.numberOfTrainingAttributes;
		if (ts.size() < 20)
			p.classNum = ts.size();
		return p;
	}
	
	
	public SVMParameter getParameters() {
		return par;
	}

	/**
	 *
	 * @return Liczba atrybutów pomniejszona o liczbę atrybutów decyzyjnych.
	 */
	public int getNumberOfTrainingAttributes() {
		return numberOfTrainingAttributes;
	}

	/**
	 * Trenuje model na podstawie danych i parametrów które są w modelu.
	 * @return SVMModel zawierający wytrenowany model.
	 */
	public SVMModel train() {
		model =new SVMModel(svm.svm_train(this, par));
		return model;
	}

	/**
	 * Trenuje model na podstawie danych i parametrów które są w modelu.
	 * @param par Zestaw parametrów dla uczenia danego modelu.
	 * @return SVMModel zawierający wytrenowany model.
	 */
	public SVMModel train(SVMParameter par) {
		return new SVMModel(svm.svm_train(this, par));	//par - tym razem z zewn�trz
	}
	
	public double crossValidation(SVMParameter par, int nr_fold, double[] target) {
		if(par.gamma ==0)
			par.gamma = 1./ numberOfTrainingAttributes;
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


	public int[][] confMatrix(SVMModel modelNu) {
		SVMNode[][] nodes = (SVMNode[][])x;
		double[] predykcje = modelNu.predict(nodes);
		for (int i=0; i<predykcje.length; i++) {
			predykcje[i] = predykcje[i] > 0 ? predykcje[i] : 0;
		}
		int[][] M =Matrix.confMatrix(y, predykcje, this.classNum);
		return M;
	}
}
