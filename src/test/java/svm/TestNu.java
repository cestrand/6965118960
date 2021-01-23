package svm;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class TestNu {
	
	public static void main(String[] args) throws Exception{
		Instances dane; 
		dane = DataSource.read("zasoby/weka-data/diabetes.arff");
		dane.setClassIndex(dane.numAttributes()-1);
		Standardize flt = new Standardize();
	//	Normalize flt = new Normalize();
		flt.setInputFormat(dane);
		dane = Filter.useFilter(dane, flt);
		
		System.out.println("A teraz tworzymy Problem typu nu-SVC:");
		SVMProblem prDia = new SVMProblem(dane); 		//dane po  Standardyzacji
	//	prDia.par.kernel_type = Parameter.POLY; //domy�lnie: Parameter.RBF
		prDia.par.svm_type = SVMParameter.NU_SVC;	//domy�lnie =3
		prDia.par.nu = 1./3;
		double[] celX = new double[dane.numInstances()];
		SVMParameter par = prDia.par;
		double[] g = {0.15};//0.1, 0.12, 0.15, 0.18, 0.2, 0.23};
		for (double gamma:g) {
			par.gamma = gamma;
			prDia.crossValidation(prDia.par,10,celX);
			System.out.println("Dla gamma = "+gamma);
		} 
		
		System.out.println("Pora na trening:");
		// ju� jest - zapisane w pliku
		prDia.par.gamma = 0.15;
		SVMModel model = prDia.train();   //to b�dzie klasyfikator
		model.saveTo("zasoby/diab-model-nu-RBF.mod");
		prDia.confusionMatrix(model);
	}
}