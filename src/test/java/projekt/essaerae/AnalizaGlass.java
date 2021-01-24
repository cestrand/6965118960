package projekt.essaerae;
import weka.SiecB;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class AnalizaGlass {

	public static void main(String[] args) throws Exception {
		Instances dane = DataSource.read("zasoby/weka-data/glass.arff");
		dane.setClassIndex(9);
		System.out.println(dane);
		SiecB bn = SiecB.createK2(2);
		bn.buildClassifier(dane);
		System.out.println(bn);
		
	}

}
