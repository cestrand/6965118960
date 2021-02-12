package svm;

import java.io.IOException;

import libsvm.svm;
import libsvm.svm_model;
import weka.core.Instance;
import weka.filters.Filter;

public class SVMModel extends svm_model{
	Filter filtr; // dla ewentualnej zmiany formatu danych

	public SVMModel(svm_model model) {
		super();
		this.param = model.param;
		this.nr_class = model.nr_class;		
		l = model.l;
		SV = model.SV;
		sv_coef = model.sv_coef;	
		rho= model.rho;		
		probA = model.probA;     
		probB = model.probB;     
		sv_indices = model.sv_indices;   
		label = model.label;
		nSV = model.nSV;
	}

	public void saveTo(String nazwaPl) throws IOException {
		svm.svm_save_model(nazwaPl, this);
	}

	public static SVMModel loadFrom(String sciezkaPliku) throws IOException {
		return new SVMModel(svm.svm_load_model(sciezkaPliku));
	}
	
	public double predict(SVMNode[] svm_nodes) {
		return svm.svm_predict(this, svm_nodes);
	}

	public double[] predict(SVMNode[][] nodes) {
		double[] predykcje = new double[nodes.length];
		for(int i = 0; i<nodes.length; i++) {
			predykcje[i] = svm.svm_predict(this, nodes[i]);
		}
		return predykcje;
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
}
