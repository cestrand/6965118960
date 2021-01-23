import java.util.Arrays;
import weka.classifiers.bayes.BayesNet;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.matrix.Matrix;
import weka.core.pmml.jaxbbindings.InstanceField;

public class Pierwsza {
	public static void main(String[] args) throws Exception {
		Instances dane = DataSource.read("zasoby/weka-data/grzyby.arff");
//		System.out.println(dane.toString());
//		System.out.println(dane.classIndex());
		dane.setClassIndex(4);
		Attribute klasa = dane.classAttribute();
//		System.out.println(klasa);
		//klasyfikacja
		BayesNet bn = new BayesNet(); // parametry s� - domy�lne
		String[] par =bn.getOptions();
    	par[5] = "2";
		bn.setOptions(par);
//		for (int i = 0; i < par.length; i++) {
//			System.out.println(i+ ": " + par[i]);
//		}
		bn.buildClassifier(dane);
		System.out.println(bn); // Stan klasyfikatora po procesie uczenia
		int N = dane.numInstances();
		for(int i = 0; i < N; i++) {
			Instance wiersz = dane.get(i);
			int dec = (int) bn.classifyInstance(wiersz);
			System.out.print(i + ": " + wiersz);
			if (dec == wiersz.classValue())
				System.out.println(" ");
			else
				System.out.println("*");
			double[] prob = bn.distributionForInstance(wiersz);
			System.out.println(Arrays.toString(prob));
		}
	}
}
