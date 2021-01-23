import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MultilayerPerceptronTest {
	
	public static void main(String[] args) throws Exception {
		Instances dane = DataSource.read("zasoby/weka-data/diabetes.arff");
		Instances daneOrg = dane; 
		daneOrg.setClassIndex(dane.numAttributes() - 1);
		MultilayerPerceptron mp = new MultilayerPerceptron();
		mp.setHiddenLayers("5,3");
		mp.setLearningRate(0.5);
		//mp.setGUI(true);
		mp.buildClassifier(daneOrg);
		System.out.println(mp);
		Evaluation ev = new Evaluation(daneOrg);
		ev.crossValidateModel(mp, daneOrg, 30, new Random(123));
		System.out.println(ev.toMatrixString());
	}
}

















