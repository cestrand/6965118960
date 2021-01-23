package svm;

import libsvm.svm_parameter;

public class SVMParameter extends svm_parameter{
	public SVMParameter() {
		// default values
		svm_type = C_SVC;
		kernel_type = RBF;
		degree = 3;
		gamma = 0;	// 1/num_features - DO ZMIANY!
		coef0 = 0;
		nu = 0.5;
		cache_size = 100;
		C = 1;
		eps = 1e-3;
		p = 0.1;
		shrinking = 1;
		probability = 0;
		nr_weight = 0;
		weight_label = new int[0];
		weight = new double[0];
//		cross_validation = 0;
		
	}

}
