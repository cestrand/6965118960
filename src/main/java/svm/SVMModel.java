package svm;

import java.io.IOException;

import libsvm.svm;
import libsvm.svm_model;
import weka.core.Instance;
import weka.filters.Filter;

public class SVMModel extends svm_model{
	Filter filtr; //docelowo - dla ewentualnej zmiany formatu nowych danych
	
	//konstruktor
	public SVMModel(svm_model model) { // stare --> nowe
		super();
		//przepisa� wszystko z modelu do wn�trza this
		this.param = model.param;
		this.nr_class = model.nr_class;		
		l = model.l;			// 
		SV = model.SV;	// 
		sv_coef = model.sv_coef;	
		rho= model.rho;		
		probA = model.probA;     
		probB = model.probB;     
		sv_indices = model.sv_indices;   
		label = model.label;		//
		nSV = model.nSV;		//
	}
			//przekazanie funkcjonalno�ci:  z klasy svm do klasy SVMModel
	public void saveTo(String nazwaPl) throws IOException {
		svm.svm_save_model(nazwaPl, this);
	}
		// jeszcze potrzeba "loadFrom()"
	
	public double predict(SVMNode[] svm_nodes) {	//wewn�trzny format danych	
		return svm.svm_predict(this, svm_nodes);
	}

	public double predict(Instance wI) {
		int m = wI.numAttributes();
		SVMNode[] svm_nodes = new SVMNode[m];
		for(int i=0; i<m; i++)
			svm_nodes[i] = new SVMNode(i+1, wI.value(i));
		return predict(svm_nodes);
	}
	public double predict(double...vals) {
		int m = vals.length;
		SVMNode[] svm_nodes = new SVMNode[m];
		for(int i=0; i<m; i++)
			svm_nodes[i] = new SVMNode(i+1, vals[i]);
		return predict(svm_nodes);
	}

	public static SVMModel fromFile(String nazwaPliku) throws IOException{
	svm_model model = svm.svm_load_model(nazwaPliku);
	return new SVMModel(model);
	}
}
